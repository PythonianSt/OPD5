import base64
import csv
import hashlib
import hmac
import html
import io
import json
import os
import secrets
import time
from datetime import date, datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response


app = FastAPI(title="KU KPS OPD Registration")
BKK = ZoneInfo("Asia/Bangkok")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.getenv("GITHUB_REPO", "").strip()
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main").strip() or "main"
CSV_PATH = os.getenv("CSV_PATH", "opd_records.csv").strip() or "opd_records.csv"
PASS_REG = os.getenv("PASS_REG", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6").strip() or "gpt-5.6"
APP_SECRET = os.getenv("APP_SECRET", "").strip() or hashlib.sha256(
    (PASS_REG or "change-me").encode("utf-8")
).hexdigest()

FACULTIES = [
    "",
    "คณะเกษตร กำแพงแสน",
    "คณะวิศวกรรมศาสตร์ กำแพงแสน",
    "คณะวิทยาศาสตร์การกีฬาและสุขภาพ",
    "คณะศิลปศาสตร์และวิทยาศาสตร์",
    "คณะศึกษาศาสตร์และพัฒนศาสตร์",
    "คณะอุตสาหกรรมบริการ",
    "คณะสัตวแพทยศาสตร์",
    "อื่นๆ",
]

FIELDS = [
    "record_id", "created_at_bkk", "updated_at_bkk", "hn",
    "first_name", "last_name", "english_name", "citizen_id",
    "student_id", "patient_type", "sex", "race", "nationality", "religion", "birth_date", "age",
    "weight", "height", "marital_status", "occupation", "faculty",
    "current_address", "phone", "email", "student_email", "emergency_contact", "emergency_phone",
    "blood_group", "treatment_right", "health_insurance",
    "chronic_disease", "drug_allergy", "food_allergy",
    "temperature", "pulse", "respiration", "blood_pressure", "spo2",
    "chief_complaint", "registration_note",
    "ai_triage_color", "ai_triage_priority", "ai_triage_chart",
    "ai_triage_discriminator", "ai_triage_rationale", "ai_triage_questions",
    "ai_triage_confidence", "ai_triage_model", "ai_triage_status",
]


def now_bkk():
    return datetime.now(BKK)


def esc(value):
    return html.escape(str(value or ""), quote=True)


def digits(value):
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def calculate_age(birth_date_text):
    try:
        born = datetime.strptime(birth_date_text, "%Y-%m-%d").date()
    except Exception:
        return ""
    today = now_bkk().date()
    if born > today:
        return ""
    return str(today.year - born.year - ((today.month, today.day) < (born.month, born.day)))


def option(value, label=None, selected=False):
    label = label if label is not None else value
    return f'<option value="{esc(value)}"{" selected" if selected else ""}>{esc(label)}</option>'


def radio(name, value, label, checked=False):
    return (
        f'<label class="radio"><input type="radio" name="{esc(name)}" '
        f'value="{esc(value)}"{" checked" if checked else ""}> {esc(label)}</label>'
    )


def combine_other(choice, other):
    choice = (choice or "").strip()
    other = (other or "").strip()
    if choice == "อื่นๆ":
        return f"อื่นๆ: {other}" if other else "อื่นๆ"
    return choice


def make_qr_data_uri(record_id):
    """Create the same simple Record-ID QR concept used in MedCertOK11.py.

    The QR contains only the record ID, not patient/clinical information.
    It is embedded directly in the success page, so showing the queue card
    does not create another GitHub API request.
    """
    try:
        import qrcode

        image = qrcode.make(str(record_id or "").strip().upper())
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        # Registration must still succeed even if the optional QR package
        # is missing; the Record ID remains usable as the queue code.
        return ""


TRIAGE_COLORS = {"RED", "ORANGE", "YELLOW", "GREEN", "BLUE", "UNDETERMINED"}


def _response_output_text(payload):
    """Extract text from an OpenAI Responses API JSON payload without SDK dependency."""
    if isinstance(payload, dict):
        direct = payload.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        for item in payload.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []) or []:
                if isinstance(content, dict) and content.get("type") == "output_text":
                    value = content.get("text", "")
                    if isinstance(value, str) and value.strip():
                        return value.strip()
    return ""


def ai_triage_suggestion(record):
    """Return MTS-informed *decision support*, never a final triage decision.

    Privacy: sends only the minimum clinical data needed for triage support.
    No name, citizen ID, student ID, phone, email, address, or record ID is sent.
    """
    if not OPENAI_API_KEY:
        return {
            "color": "UNDETERMINED", "priority": "", "chart": "",
            "discriminator": "", "rationale": "ไม่ได้ตั้ง OPENAI_API_KEY",
            "questions": [], "confidence": "LOW", "status": "AI_NOT_CONFIGURED",
        }

    clinical = {
        "age": record.get("age", ""),
        "sex": record.get("sex", ""),
        "chief_complaint": record.get("chief_complaint", ""),
        "temperature_c": record.get("temperature", ""),
        "pulse_per_min": record.get("pulse", ""),
        "respiration": record.get("respiration", ""),
        "blood_pressure_mmHg": record.get("blood_pressure", ""),
        "spo2_percent": record.get("spo2", ""),
        "chronic_disease": record.get("chronic_disease", ""),
        "drug_allergy": record.get("drug_allergy", ""),
    }

    instructions = """You are a clinical triage decision-support assistant for a university outpatient clinic.
Use Manchester Triage System (MTS) principles only as a reference framework, not as an autonomous final decision.
Core method: identify the most specific presenting complaint/chart; consider the highest-order applicable discriminator first; prioritize life threat, consciousness, haemorrhage, temperature, pain, and acuteness; use vital signs only where clinically relevant; do not diagnose disease from triage data.
The source data are incomplete self-reported symptoms plus vital signs. MTS is a face-to-face clinical system, so if required visual examination, focused questions, pain assessment, mechanism, mental-health risk, or other discriminator data are missing, say so and return questions_for_nurse.
Never invent a finding. Never downgrade because a vital sign is normal when the complaint could contain a higher-order discriminator not yet assessed.
Return a suggested color only when reasonably supported. If data are insufficient to distinguish priorities safely, use UNDETERMINED.
Colors: RED=priority 1 immediate, ORANGE=priority 2 very urgent, YELLOW=priority 3 urgent, GREEN=priority 4 standard, BLUE=priority 5 non-urgent, UNDETERMINED=needs clinician assessment.
This output is advisory; a trained nurse/clinician must confirm or override it after direct assessment.
Respond in Thai, concise, and return JSON exactly matching the requested schema."""

    schema = {
        "type": "object",
        "properties": {
            "color": {"type": "string", "enum": ["RED", "ORANGE", "YELLOW", "GREEN", "BLUE", "UNDETERMINED"]},
            "priority": {"type": "string"},
            "suggested_chart": {"type": "string"},
            "possible_discriminator": {"type": "string"},
            "rationale_th": {"type": "string"},
            "questions_for_nurse": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
            "confidence": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
        },
        "required": ["color", "priority", "suggested_chart", "possible_discriminator", "rationale_th", "questions_for_nurse", "confidence"],
        "additionalProperties": False,
    }

    body = {
        "model": OPENAI_MODEL,
        "instructions": instructions,
        "input": [{
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": "Clinical data for pre-triage decision support (JSON):\n" + json.dumps(clinical, ensure_ascii=False),
            }],
        }],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "mts_pretriage_suggestion",
                "strict": True,
                "schema": schema,
            }
        },
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=25,
        )
        response.raise_for_status()
        raw = _response_output_text(response.json())
        result = json.loads(raw)
        color = str(result.get("color", "UNDETERMINED")).upper().strip()
        if color not in TRIAGE_COLORS:
            color = "UNDETERMINED"
        questions = result.get("questions_for_nurse", [])
        if not isinstance(questions, list):
            questions = []
        return {
            "color": color,
            "priority": str(result.get("priority", "")),
            "chart": str(result.get("suggested_chart", "")),
            "discriminator": str(result.get("possible_discriminator", "")),
            "rationale": str(result.get("rationale_th", "")),
            "questions": [str(x) for x in questions[:4]],
            "confidence": str(result.get("confidence", "LOW")),
            "status": "OK",
        }
    except requests.HTTPError as error:
        status = getattr(getattr(error, "response", None), "status_code", "")
        return {
            "color": "UNDETERMINED", "priority": "", "chart": "",
            "discriminator": "", "rationale": f"AI ไม่พร้อมใช้งาน (HTTP {status})",
            "questions": [], "confidence": "LOW", "status": "AI_HTTP_ERROR",
        }
    except Exception as error:
        return {
            "color": "UNDETERMINED", "priority": "", "chart": "",
            "discriminator": "", "rationale": f"AI ไม่พร้อมใช้งาน ({type(error).__name__})",
            "questions": [], "confidence": "LOW", "status": "AI_ERROR",
        }


def triage_badge_html(record):
    color = str(record.get("ai_triage_color", "UNDETERMINED") or "UNDETERMINED").upper()
    labels = {
        "RED": "🔴 RED", "ORANGE": "🟠 ORANGE", "YELLOW": "🟡 YELLOW",
        "GREEN": "🟢 GREEN", "BLUE": "🔵 BLUE", "UNDETERMINED": "⚪ ต้องประเมินเพิ่ม",
    }
    bg = {
        "RED": "#fef3f2", "ORANGE": "#fff4ed", "YELLOW": "#fffaeb",
        "GREEN": "#ecfdf3", "BLUE": "#eff8ff", "UNDETERMINED": "#f2f4f7",
    }.get(color, "#f2f4f7")
    fg = {
        "RED": "#b42318", "ORANGE": "#b54708", "YELLOW": "#7a2e0e",
        "GREEN": "#067647", "BLUE": "#175cd3", "UNDETERMINED": "#475467",
    }.get(color, "#475467")
    return f'<span style="display:inline-block;padding:4px 9px;border-radius:999px;background:{bg};color:{fg};font-weight:800">{esc(labels.get(color, color))}</span>'


def github_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def github_url():
    encoded_path = quote(CSV_PATH, safe="/")
    return f"https://api.github.com/repos/{GITHUB_REPO}/contents/{encoded_path}"


def check_server_config():
    missing = []
    if not GITHUB_TOKEN:
        missing.append("GITHUB_TOKEN")
    if not GITHUB_REPO:
        missing.append("GITHUB_REPO")
    if not PASS_REG:
        missing.append("PASS_REG")
    return missing


def read_records():
    missing = check_server_config()
    if missing:
        raise RuntimeError("ยังไม่ได้ตั้ง Environment Variables: " + ", ".join(missing))

    response = requests.get(
        github_url(),
        headers=github_headers(),
        params={"ref": GITHUB_BRANCH},
        timeout=20,
    )
    if response.status_code == 404:
        return [], None
    response.raise_for_status()
    payload = response.json()
    content = base64.b64decode(payload["content"]).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    rows = []
    for row in reader:
        normalized = {field: str(row.get(field, "") or "") for field in FIELDS}
        rows.append(normalized)
    return rows, payload.get("sha")


def serialize_records(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in FIELDS})
    return buffer.getvalue()


def write_records(rows, sha=None, message="update OPD records"):
    csv_text = serialize_records(rows)
    encoded = base64.b64encode(("\ufeff" + csv_text).encode("utf-8")).decode("ascii")
    payload = {
        "message": message,
        "content": encoded,
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    response = requests.put(
        github_url(),
        headers=github_headers(),
        json=payload,
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def github_error_text(error):
    """Return a short, useful GitHub HTTP error without exposing secrets."""
    response = getattr(error, "response", None)
    if response is None:
        return f"{type(error).__name__}: ไม่ได้รับคำตอบจาก GitHub"
    status = response.status_code
    try:
        message = str(response.json().get("message", "")).strip()
    except Exception:
        message = ""
    remaining = response.headers.get("X-RateLimit-Remaining")
    reset_at = response.headers.get("X-RateLimit-Reset")
    retry_after = response.headers.get("Retry-After")
    details = [f"HTTP {status}"]
    if message:
        details.append(message)
    if remaining is not None:
        details.append(f"RateLimit-Remaining={remaining}")
    if retry_after:
        details.append(f"Retry-After={retry_after}s")
    if reset_at:
        try:
            reset_text = datetime.fromtimestamp(int(reset_at), tz=BKK).strftime("%d/%m/%Y %H:%M:%S")
            details.append(f"Reset={reset_text}")
        except Exception:
            pass
    return " | ".join(details)


def retry_delay(error, attempt):
    """Back off for conflict/rate-limit responses. Returns seconds or None."""
    response = getattr(error, "response", None)
    if response is None:
        return None
    status = response.status_code
    if status in (409, 422):
        return 0.35 * (attempt + 1)
    if status in (403, 429):
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(max(float(retry_after), 0.5), 8.0)
            except Exception:
                pass
        return min(1.0 * (2 ** attempt), 8.0)
    return None


def append_record(record):
    # Retry if two people submit together (SHA conflict) or GitHub asks us to slow down.
    last_error = None
    for attempt in range(4):
        try:
            rows, sha = read_records()
            if any(row.get("record_id") == record["record_id"] for row in rows):
                return
            rows.append(record)
            write_records(
                rows,
                sha,
                message=f"add OPD record {record['record_id']} {record['created_at_bkk']}",
            )
            return
        except requests.HTTPError as error:
            last_error = error
            delay = retry_delay(error, attempt)
            if delay is not None and attempt < 3:
                time.sleep(delay)
                continue
            raise
    if last_error:
        raise last_error


def update_record(record_id, updates):
    last_error = None
    for attempt in range(4):
        try:
            rows, sha = read_records()
            found = False
            for row in rows:
                if row.get("record_id") == record_id:
                    row.update(updates)
                    row["updated_at_bkk"] = now_bkk().isoformat()
                    found = True
                    break
            if not found:
                return False
            write_records(rows, sha, message=f"update OPD record {record_id}")
            return True
        except requests.HTTPError as error:
            last_error = error
            delay = retry_delay(error, attempt)
            if delay is not None and attempt < 3:
                time.sleep(delay)
                continue
            raise
    if last_error:
        raise last_error
    return False


def find_record(record_id):
    rows, _ = read_records()
    for row in rows:
        if row.get("record_id") == record_id:
            return row
    return None


def auth_token():
    return hmac.new(APP_SECRET.encode(), b"registry", hashlib.sha256).hexdigest()


def is_registry_authenticated(request):
    token = request.cookies.get("opd_registry_auth", "")
    return bool(token) and hmac.compare_digest(token, auth_token())


def html_page(title, body, extra_head=""):
    return f"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>{esc(title)}</title>
<style>
:root {{
  --bg:#f5f7fa; --card:#fff; --ink:#172033; --muted:#667085;
  --line:#d0d5dd; --primary:#175cd3; --danger:#b42318; --ok:#067647;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI","Noto Sans Thai",Tahoma,sans-serif;
}}
.wrap {{ max-width:980px; margin:0 auto; padding:18px; }}
.card {{
  background:var(--card); border:1px solid #e4e7ec; border-radius:14px;
  padding:20px; margin-bottom:16px; box-shadow:0 1px 2px rgba(16,24,40,.04);
}}
h1 {{ font-size:1.55rem; margin:0 0 8px; }}
h2 {{ font-size:1.15rem; margin:4px 0 14px; }}
h3 {{ font-size:1rem; margin:12px 0 8px; }}
p.note {{ color:var(--muted); margin-top:4px; }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px 16px; }}
.grid3 {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
.full {{ grid-column:1/-1; }}
label.field {{ display:block; font-weight:600; font-size:.92rem; }}
input, select, textarea {{
  width:100%; padding:10px 11px; margin-top:6px; border:1px solid var(--line);
  border-radius:9px; font:inherit; background:#fff; color:var(--ink);
}}
textarea {{ min-height:90px; resize:vertical; }}
.radios {{ display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:7px; }}
.radio {{ font-weight:400; white-space:nowrap; }}
.radio input {{ width:auto; margin:0 4px 0 0; }}
button,.btn {{
  appearance:none; border:0; border-radius:9px; padding:10px 16px;
  font:inherit; font-weight:700; cursor:pointer; background:var(--primary); color:white;
  text-decoration:none; display:inline-block;
}}
.btn.secondary {{ background:#475467; }}
.btn.light {{ color:#344054; background:#eef2f6; }}
.btn.danger {{ background:var(--danger); }}
.actions {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
.alert {{ padding:12px 14px; border-radius:9px; margin:12px 0; }}
.alert.ok {{ background:#ecfdf3; color:var(--ok); }}
.alert.err {{ background:#fef3f2; color:var(--danger); }}
.mono {{
  width:100%; min-height:330px; white-space:pre-wrap; background:#101828; color:#f2f4f7;
  border-radius:10px; padding:14px; font-family:ui-monospace,SFMono-Regular,Consolas,monospace;
  font-size:.9rem; overflow:auto;
}}
table {{ width:100%; border-collapse:collapse; }}
th,td {{ text-align:left; padding:9px 8px; border-bottom:1px solid #eaecf0; vertical-align:top; }}
th {{ font-size:.88rem; color:#475467; }}
a {{ color:#175cd3; }}
.small {{ font-size:.88rem; color:var(--muted); }}
.hidden {{ display:none; }}
.a4 {{
  width:210mm; min-height:297mm; max-width:100%;
  background:white; color:#111; margin:0 auto; padding:12mm 14mm;
  border:1px solid #ddd;
}}
.a4 h1 {{ text-align:center; font-size:20pt; }}
.a4 h2 {{ font-size:14pt; border-bottom:1px solid #555; padding-bottom:3px; }}
.a4 p,.a4 td,.a4 th {{ font-size:10.2pt; line-height:1.22; }}
.a4 table {{ margin-bottom:5px; }}
.a4 th,.a4 td {{ padding:5px 6px; }}
.a4 h2 {{ margin:3mm 0 1.5mm; }}
.complaint-box {{ white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; min-height:10mm; }}
.print-page {{ break-after: page; page-break-after: always; }}
.print-page:last-child {{ break-after: auto; page-break-after: auto; }}

/* A4 sheet containing two identical half-A4 certificates */
.certificate-a4-sheet {{
  width:210mm;
  height:297mm;
  max-width:100%;
  background:#fff;
  color:#111;
  margin:0 auto;
  padding:0;
}}
.medcert {{
  box-sizing:border-box;
  width:210mm;
  height:148.5mm;
  max-width:100%;
  overflow:hidden;
  background:white;
  color:#111;
  margin:0;
  padding:6mm 10mm;
  border:1px solid #ddd;
  font-family:"Noto Sans Thai",Tahoma,sans-serif;
}}
.medcert-copy-1, .medcert-copy-2 {{
  width:210mm;
  height:148.5mm;
  margin:0;
  padding:0;
}}
.medcert-copy-1 {{
  border-bottom:1px dashed #777;
}}
.medcert h1 {{
  text-align:center;
  font-size:16.5pt;
  line-height:1.05;
  margin:0 0 1.5mm;
}}
.medcert .center {{
  text-align:center;
}}
.medcert p {{
  font-size:10.8pt;
  line-height:1.28;
  margin:0.7mm 0;
}}
.medcert .line {{
  display:inline-block;
  border-bottom:1px dotted #333;
  min-width:35mm;
  min-height:1em;
  vertical-align:baseline;
  padding:0 1mm;
}}
.medcert .line-long {{ min-width:74mm; }}
.medcert .line-short {{ min-width:23mm; }}
.medcert .signature-grid {{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:10mm;
  margin-top:4mm;
}}
.medcert .signature-box {{
  text-align:center;
  font-size:10.2pt;
  line-height:1.25;
}}
.medcert .footer-note {{
  margin-top:3mm;
  font-size:8.7pt;
  line-height:1.2;
}}
.cut-note {{
  position:relative;
  height:0;
  text-align:center;
  font-size:8pt;
  color:#666;
  z-index:2;
}}
.queue-ticket {{
  width:92mm; max-width:100%; margin:18px auto; padding:8mm 7mm;
  background:#fff; border:2px solid #172033; border-radius:12px; text-align:center;
}}
.queue-ticket h2 {{ margin:0 0 3mm; font-size:18pt; }}
.queue-ticket .queue-id {{ font-size:24pt; font-weight:900; letter-spacing:1.5px; margin:2mm 0; }}
.queue-ticket .queue-meta {{ font-size:11pt; line-height:1.45; margin:1mm 0; }}
.queue-ticket .qr {{ width:42mm; height:42mm; max-width:78%; object-fit:contain; margin:3mm auto; display:block; }}
.queue-ticket .queue-note {{ font-size:9.5pt; color:#475467; margin-top:2mm; }}

@media(max-width:720px) {{
  .grid,.grid3 {{ grid-template-columns:1fr; }}
  .wrap {{ padding:10px; }}
  .card {{ padding:15px; }}
}}
@media print {{
  body {{ background:white; }}
  .no-print {{ display:none !important; }}
  .wrap {{ max-width:none; padding:0; }}
  .card {{ border:0; box-shadow:none; padding:0; margin:0; }}
  .a4 {{ border:0; width:210mm; min-height:0; height:auto; padding:7mm 9mm; break-inside:auto; page-break-inside:auto; }}
  .a4 h1 {{ font-size:17pt; margin-bottom:2mm; }}
  .a4 h2 {{ font-size:11.5pt; margin:2.2mm 0 1mm; }}
  .a4 p,.a4 td,.a4 th {{ font-size:9.2pt; line-height:1.15; }}
  .a4 th,.a4 td {{ padding:3px 4px; }}
  .certificate-a4-sheet {{ width:210mm; height:297mm; margin:0; padding:0; }}
  .medcert {{
    border-left:0;
    border-right:0;
    width:210mm;
    height:148.5mm;
    min-height:148.5mm;
    padding:6mm 10mm;
  }}
  .medcert-copy-1, .medcert-copy-2 {{
    width:210mm;
    height:148.5mm;
    overflow:hidden;
  }}
  .medcert-copy-1 {{ border-top:0; border-bottom:1px dashed #777; }}
  .medcert-copy-2 {{ border-bottom:0; }}
  .queue-ticket {{
    width:92mm; margin:12mm auto 0; padding:7mm; border:1.5px solid #111;
    page-break-inside:avoid; break-inside:avoid;
  }}
  .queue-only-print {{ display:block !important; }}
  @page {{ size:A4 portrait; margin:0; }}
}}
</style>
{extra_head}
</head>
<body>{body}</body></html>"""


def patient_form(error=""):
    faculty_options = "".join(option(x, "— เลือกคณะ —" if x == "" else x) for x in FACULTIES)
    body = f"""
<div class="wrap">
  <div class="card">
    <h1>ลงทะเบียนผู้รับบริการ OPD</h1>
    <p class="note">สถานพยาบาลมหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตกำแพงแสน</p>
    <div class="alert" style="background:#fff7ed;color:#9a3412;border:1px solid #fed7aa;font-weight:700">
      * เพื่อประโยชน์และความปลอดภัยของท่าน โปรดกรอกข้อมูลให้ครบถ้วนและลงลายชื่อกำกับ *
    </div>
    {f'<div class="alert err">{esc(error)}</div>' if error else ''}
    <form method="post" action="/submit" autocomplete="on">
      <div id="personal_details"><h2>1. ข้อมูลส่วนบุคคล</h2>
      <div class="grid">
        <label class="field">ชื่อ *<input name="first_name" required></label>
        <label class="field">นามสกุล *<input name="last_name" required></label>
        <label class="field full">ชื่อภาษาอังกฤษ<input name="english_name" placeholder="Name Surname"></label>
        <label class="field">เลขบัตรประชาชน 13 หลัก *
          <input name="citizen_id" inputmode="numeric" maxlength="13" pattern="[0-9]{{13}}" required>
        </label>
        <label class="field">รหัสนิสิต
          <input name="student_id" id="student_id" inputmode="numeric" placeholder="กรอกสำหรับนิสิต">
        </label>
        <div class="field full"><b>ประเภทผู้รับบริการ *</b><div class="radios">
          {radio("patient_type","คนไข้ใหม่","คนไข้ใหม่")}
          {radio("patient_type","คนไข้เก่า","คนไข้เก่า")}
        </div></div>
        <div id="old_patient_note" class="alert hidden" style="background:#eff8ff;color:#175cd3">
          คนไข้เก่า: กรุณากรอกเลขบัตรประชาชนและอาการนำเป็นหลัก ข้อมูลอื่นกรอกเฉพาะรายการที่ต้องการ Update เช่น เบอร์โทรศัพท์หรือที่อยู่
        </div>
        <div class="field"><b>เพศ</b><div class="radios">
          {radio("sex","หญิง","หญิง",True)}
          {radio("sex","ชาย","ชาย")}
          {radio("sex","ไม่ระบุ","ไม่ระบุ")}
        </div></div>

        <div class="field"><b>เชื้อชาติ</b><div class="radios">
          {radio("race_choice","ไทย","ไทย",True)}
          {radio("race_choice","อื่นๆ","อื่นๆ")}
        </div><input name="race_other" id="race_other" class="hidden" placeholder="ระบุเชื้อชาติ"></div>

        <div class="field"><b>สัญชาติ</b><div class="radios">
          {radio("nationality_choice","ไทย","ไทย",True)}
          {radio("nationality_choice","อื่นๆ","อื่นๆ")}
        </div><input name="nationality_other" id="nationality_other" class="hidden" placeholder="ระบุสัญชาติ"></div>

        <div class="field"><b>ศาสนา</b><div class="radios">
          {radio("religion_choice","พุทธ","พุทธ",True)}
          {radio("religion_choice","อื่นๆ","อื่นๆ")}
        </div><input name="religion_other" id="religion_other" class="hidden" placeholder="ระบุศาสนา"></div>

        <label class="field">วันเกิด *<input type="date" name="birth_date" id="birth_date" required></label>
        <label class="field">อายุ (คำนวณอัตโนมัติ)<input id="age_preview" readonly></label>
        <label class="field">น้ำหนัก (กก.)<input name="weight" inputmode="decimal"></label>
        <label class="field">ส่วนสูง (ซม.)<input name="height" inputmode="decimal"></label>

        <div class="field"><b>สถานภาพ</b><div class="radios">
          {radio("marital_choice","โสด","โสด",True)}
          {radio("marital_choice","สมรส","สมรส")}
          {radio("marital_choice","หย่า","หย่า")}
          {radio("marital_choice","อื่นๆ","อื่นๆ")}
        </div><input name="marital_other" id="marital_other" class="hidden" placeholder="ระบุสถานภาพ"></div>

        <div class="field"><b>อาชีพ</b><div class="radios">
          {radio("occupation_choice","นักศึกษา","นักศึกษา",True)}
          {radio("occupation_choice","บุคลากร","บุคลากร")}
          {radio("occupation_choice","บุคคลทั่วไป","บุคคลทั่วไป")}
          {radio("occupation_choice","อื่นๆ","อื่นๆ")}
        </div><input name="occupation_other" id="occupation_other" class="hidden" placeholder="ระบุอาชีพ"></div>

        <label class="field full">คณะ
          <select name="faculty" id="faculty">{faculty_options}</select>
          <input name="faculty_other" id="faculty_other" class="hidden" placeholder="กรุณาระบุคณะ">
        </label>
      </div></div>

      <div id="contact_details"><h2 style="margin-top:24px">2. ข้อมูลติดต่อ</h2>
      <div class="grid">
        <label class="field full">ที่อยู่ปัจจุบัน<textarea name="current_address"></textarea></label>
        <label class="field">เบอร์โทรศัพท์ผู้รับบริการ<input name="phone" inputmode="tel"></label>
        <label class="field">อีเมลผู้รับบริการ<input type="email" name="email" placeholder="name@example.com"></label>
        <label class="field">อีเมลนิสิตมหาวิทยาลัย (@ku.th)
          <input type="email" name="student_email" id="student_email" placeholder="เช่น b69xxxxxxxx@ku.th">
        </label>
        <label class="field">ผู้ติดต่อกรณีฉุกเฉิน<input name="emergency_contact"></label>
        <label class="field">เบอร์โทรศัพท์ผู้ติดต่อกรณีฉุกเฉิน<input name="emergency_phone" inputmode="tel"></label>
      </div></div>

      <div id="health_details"><h2 style="margin-top:24px">3. ประวัติสุขภาพ</h2>
      <div class="grid">
        <label class="field">กรุ๊ปเลือด
          <select name="blood_group">
            <option value="">— เลือก —</option><option>O</option><option>A</option><option>B</option><option>AB</option>
          </select>
        </label>
        <div class="field"><b>สิทธิการรักษา</b><div class="radios">
          {radio("treatment_right","ไม่มี","ไม่มี",True)}
          {radio("treatment_right","มี","มี")}
        </div></div>
        <div class="field"><b>ประกันสุขภาพ</b><div class="radios">
          {radio("health_insurance","ไม่มี","ไม่มี",True)}
          {radio("health_insurance","มี","มี")}
        </div></div>
        <label class="field full">โรคประจำตัว<input name="chronic_disease" value="ไม่มี"></label>
        <label class="field full">ประวัติแพ้ยา<input name="drug_allergy" value="ไม่มี"></label>
        <label class="field full">ประวัติแพ้อาหาร<input name="food_allergy" value="ไม่มี"></label>
      </div></div>

      <div id="vitals_details"><h2 style="margin-top:24px">4. สัญญาณชีพ</h2>
      <p class="small">วันและเวลารับบริการจะบันทึกอัตโนมัติตามเขตเวลา Asia/Bangkok เมื่อกดบันทึก</p>
      <div class="grid3">
        <label class="field">อุณหภูมิ (°C)<input name="temperature" inputmode="decimal"></label>
        <label class="field">ชีพจร (ครั้ง/นาที)<input name="pulse" inputmode="numeric"></label>
        <label class="field">การหายใจ<input name="respiration" value="ปกติ"></label>
        <label class="field">ความดันโลหิต<input name="blood_pressure" placeholder="120/80"></label>
        <label class="field">ออกซิเจนปลายนิ้ว SpO₂ (%)<input name="spo2" inputmode="numeric"></label>
      </div></div>

      <h2 style="margin-top:24px">5. อาการนำ (ไม่สบาย เป็นอะไร)</h2>
      <label class="field">อาการนำ (ไม่สบาย เป็นอะไร)<textarea name="chief_complaint" placeholder="กรุณาระบุว่าไม่สบาย เป็นอะไร"></textarea></label>

      <div style="height:14px"></div>
      <label class="radio"><input type="checkbox" required> ข้าพเจ้าตรวจสอบข้อมูลว่าถูกต้องและครบถ้วนก่อนส่งแล้ว</label>
      <div style="height:14px"></div>
      <button type="submit">บันทึกข้อมูล OPD</button>
    </form>
  </div>
</div>
<script>

function refreshPatientType() {{
  const chosen = document.querySelector('input[name="patient_type"]:checked');
  const isOld = chosen && chosen.value === "คนไข้เก่า";
  const note = document.getElementById("old_patient_note");
  if (note) note.classList.toggle("hidden", !isOld);

  // For old patients, the registry will retrieve the paper chart by citizen ID.
  // Keep all fields available for optional updates, but remove mandatory validation.
  ["first_name","last_name","birth_date"].forEach(name => {{
    const el = document.querySelector(`[name="${{name}}"]`);
    if (el) el.required = !isOld;
  }});

  // Student ID / KU email are required only for NEW student registrations.
  refreshStudentFields();
}}

document.querySelectorAll('input[name="patient_type"]').forEach(x => x.addEventListener("change", refreshPatientType));

function bindOther(radioName, inputId) {{
  const items = document.querySelectorAll(`input[name="${{radioName}}"]`);
  const box = document.getElementById(inputId);
  function refresh() {{
    const chosen = document.querySelector(`input[name="${{radioName}}"]:checked`);
    const show = chosen && chosen.value === "อื่นๆ";
    box.classList.toggle("hidden", !show);
    if (!show) box.value = "";
  }}
  items.forEach(x => x.addEventListener("change", refresh));
  refresh();
}}
bindOther("race_choice","race_other");
bindOther("nationality_choice","nationality_other");
bindOther("religion_choice","religion_other");
bindOther("marital_choice","marital_other");
bindOther("occupation_choice","occupation_other");

function refreshStudentFields() {{
  const occupation = document.querySelector('input[name="occupation_choice"]:checked');
  const isStudent = occupation && occupation.value === "นักศึกษา";
  const patientType = document.querySelector('input[name="patient_type"]:checked');
  const isOld = patientType && patientType.value === "คนไข้เก่า";
  const sid = document.getElementById("student_id");
  const semail = document.getElementById("student_email");
  sid.required = isStudent && !isOld;
  semail.required = isStudent && !isOld;
}}
document.querySelectorAll('input[name="occupation_choice"]').forEach(x => x.addEventListener("change", refreshStudentFields));
refreshStudentFields();
refreshPatientType();

const facultySelect = document.getElementById("faculty");
const facultyOther = document.getElementById("faculty_other");
function refreshFacultyOther() {{
  const show = facultySelect && facultySelect.value === "อื่นๆ";
  facultyOther.classList.toggle("hidden", !show);
  facultyOther.required = !!show;
  if (!show) facultyOther.value = "";
}}
if (facultySelect) {{
  facultySelect.addEventListener("change", refreshFacultyOther);
  refreshFacultyOther();
}}

document.getElementById("birth_date").addEventListener("change", function() {{
  const d = new Date(this.value + "T00:00:00");
  if (isNaN(d)) return document.getElementById("age_preview").value = "";
  const t = new Date();
  let age = t.getFullYear() - d.getFullYear();
  const m = t.getMonth() - d.getMonth();
  if (m < 0 || (m === 0 && t.getDate() < d.getDate())) age--;
  document.getElementById("age_preview").value = age >= 0 ? age : "";
}});
</script>
"""
    return html_page("ลงทะเบียน OPD", body)


def record_text(row):
    return f"""ข้อมูลผู้รับบริการ OPD
Record ID: {row.get('record_id','')}
HN: {row.get('hn','')}
วันเวลา: {format_datetime(row.get('created_at_bkk',''))}

ข้อมูลส่วนบุคคล
ชื่อ: {row.get('first_name','')}
นามสกุล: {row.get('last_name','')}
ชื่อภาษาอังกฤษ: {row.get('english_name','')}
บัตรประชาชน: {row.get('citizen_id','')}
รหัสนิสิต: {row.get('student_id','')}
ประเภทผู้รับบริการ: {row.get('patient_type','')}
เพศ: {row.get('sex','')}
เชื้อชาติ: {row.get('race','')}
สัญชาติ: {row.get('nationality','')}
ศาสนา: {row.get('religion','')}
วันเกิด: {format_date(row.get('birth_date',''))}
อายุ: {row.get('age','')} ปี
น้ำหนัก: {row.get('weight','')} กก.
ส่วนสูง: {row.get('height','')} ซม.
สถานภาพ: {row.get('marital_status','')}
อาชีพ: {row.get('occupation','')}
คณะ: {row.get('faculty','')}

ข้อมูลติดต่อ
ที่อยู่ปัจจุบัน: {row.get('current_address','')}
โทรศัพท์: {row.get('phone','')}
อีเมลผู้รับบริการ: {row.get('email','')}
อีเมลนิสิตมหาวิทยาลัย: {row.get('student_email','')}
ผู้ติดต่อกรณีฉุกเฉิน: {row.get('emergency_contact','')}
โทรศัพท์ผู้ติดต่อกรณีฉุกเฉิน: {row.get('emergency_phone','')}

ประวัติสุขภาพ
กรุ๊ปเลือด: {row.get('blood_group','')}
สิทธิการรักษา: {row.get('treatment_right','')}
ประกันสุขภาพ: {row.get('health_insurance','')}
โรคประจำตัว: {row.get('chronic_disease','')}
ประวัติแพ้ยา: {row.get('drug_allergy','')}
ประวัติแพ้อาหาร: {row.get('food_allergy','')}

สัญญาณชีพ
T: {row.get('temperature','')} °C
P: {row.get('pulse','')} /min
R: {row.get('respiration','')}
BP: {row.get('blood_pressure','')} mmHg
SpO2: {row.get('spo2','')} %

อาการนำ
{row.get('chief_complaint','')}

AI Pre-triage (อ้างอิงแนวทาง MTS; ต้องให้พยาบาล/แพทย์ยืนยัน)
สีที่แนะนำ: {row.get('ai_triage_color','')} {row.get('ai_triage_priority','')}
Presentation chart: {row.get('ai_triage_chart','')}
Possible discriminator: {row.get('ai_triage_discriminator','')}
เหตุผล: {row.get('ai_triage_rationale','')}
คำถามเพิ่มเติม: {row.get('ai_triage_questions','')}
Confidence: {row.get('ai_triage_confidence','')}

หมายเหตุเวชระเบียน
{row.get('registration_note','')}
""".strip()


def format_date(value):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").strftime("%d/%m/%Y")
    except Exception:
        return str(value or "")


def format_datetime(value):
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=BKK)
        parsed = parsed.astimezone(BKK)
        return parsed.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return str(value or "")



def thai_date_from_timestamp(value):
    """Return Thai Buddhist-calendar date from a Bangkok/ISO timestamp."""
    raw = str(value or "").strip()
    parsed = None

    if raw:
        # Accept normal ISO timestamps and timestamps ending in Z.
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            pass

        # Also tolerate common date/time strings if data came from an older CSV.
        if parsed is None:
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
                "%d/%m/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M",
                "%d/%m/%Y",
            ):
                try:
                    parsed = datetime.strptime(raw, fmt)
                    break
                except Exception:
                    continue

    # The issue date must not disappear on the printed certificate.
    # created_at_bkk is used by the caller; this fallback is only for malformed/legacy rows.
    if parsed is None:
        parsed = now_bkk()

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=BKK)
    parsed = parsed.astimezone(BKK)

    months = [
        "", "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
        "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
    ]
    return f"{parsed.day} {months[parsed.month]} {parsed.year + 543}"


def medical_certificate_content(row):
    issue_timestamp = (
        row.get("created_at_bkk", "")
        or row.get("updated_at_bkk", "")
        or now_bkk().isoformat()
    )
    issue_date = thai_date_from_timestamp(issue_timestamp)
    patient_name = f"{row.get('first_name','')} {row.get('last_name','')}".strip()
    citizen_id = row.get("citizen_id", "")
    return f"""
<div class="medcert">
  <h1>ใบรับรองแพทย์</h1>
  <p class="center">
    <b>สถานพยาบาลมหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตกำแพงแสน</b><br>
    เลขที่ 1 หมู่ 6 ต.กำแพงแสน อ.กำแพงแสน จ.นครปฐม 73140
  </p>
  <p style="text-align:right"><b>วันที่</b> <span class="line">{esc(issue_date)}</span></p>

  <p>
    ข้าพเจ้า <span class="line">นายแพทย์กำธร ตันติวิทยาทันต์</span>
    ได้รับใบอนุญาตประกอบวิชาชีพเวชกรรมเลขที่
    <span class="line line-short">12082</span>
  </p>

  <p>
    ได้ทำการตรวจร่างกายของผู้ป่วยชื่อ
    <span class="line line-long">{esc(patient_name)}</span>
    ทะเบียนผู้ป่วยเลขที่ <span class="line">&nbsp;</span>
  </p>

  <p>
    เลขประจำตัวประชาชน
    <span class="line line-long">{esc(citizen_id)}</span>
  </p>

  <p>
    มีอาการ / การวินิจฉัยโรค
    <span class="line line-long">&nbsp;</span>
  </p>

  <p>
    มีความเห็น
    <span class="line line-long">&nbsp;</span>
  </p>

  <p>
    ( &nbsp; ) ได้รับการตรวจจริง เมื่อวันที่
    <span class="line line-short">&nbsp;</span>
    เดือน <span class="line line-short">&nbsp;</span>
    พ.ศ. <span class="line line-short">&nbsp;</span>
  </p>

  <p>
    ( &nbsp; ) สมควรพักตั้งแต่ วันที่
    <span class="line line-short">&nbsp;</span>
    เดือน <span class="line line-short">&nbsp;</span>
    พ.ศ. <span class="line line-short">&nbsp;</span>
  </p>

  <p>
    ถึง วันที่
    <span class="line line-short">&nbsp;</span>
    เดือน <span class="line line-short">&nbsp;</span>
    พ.ศ. <span class="line line-short">&nbsp;</span>
    รวม <span class="line line-short">&nbsp;</span> วัน
  </p>

  <p style="margin-top:8mm">
    ทั้งนี้ได้ให้การรักษาพร้อมด้วยคำแนะนำไว้เรียบร้อยแล้ว
  </p>

  <div class="signature-grid">
    <div class="signature-box">
      ลงชื่อ ........................................................<br>
      ผู้เข้ารับการตรวจ
    </div>
    <div class="signature-box">
      ลงชื่อ ........................................................<br>
      (นายแพทย์กำธร ตันติวิทยาทันต์)<br>
      แพทย์ผู้ตรวจ
    </div>
  </div>

  <p class="footer-note">
    <b>หมายเหตุ :</b> ใบรับรองแพทย์นี้จะสมบูรณ์เมื่อมีการประทับตรา
    สถานพยาบาลมหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตกำแพงแสน เท่านั้น
  </p>
</div>
"""


def a4_content(row):
    def tr(label, value):
        return f"<tr><th>{esc(label)}</th><td>{esc(value)}</td></tr>"

    complaint = esc(row.get("chief_complaint", "")).replace(chr(10), "<br>") or "&nbsp;"
    note = esc(row.get("registration_note", "")).replace(chr(10), "<br>") or "&nbsp;"
    is_old = row.get("patient_type", "") == "คนไข้เก่า"

    if is_old:
        # Existing patients: print only what the patient entered/updated, plus complaint.
        optional_rows = []
        printable = [
            ("ชื่อ", f"{row.get('first_name','')} {row.get('last_name','')}".strip()),
            ("ชื่อภาษาอังกฤษ", row.get("english_name", "")),
            ("รหัสนิสิต", row.get("student_id", "")),
            ("เพศ", row.get("sex", "") if row.get("sex", "") not in ("", "หญิง") else ""),
            ("เชื้อชาติ", row.get("race", "") if row.get("race", "") not in ("", "ไทย") else ""),
            ("สัญชาติ", row.get("nationality", "") if row.get("nationality", "") not in ("", "ไทย") else ""),
            ("ศาสนา", row.get("religion", "") if row.get("religion", "") not in ("", "พุทธ") else ""),
            ("วันเกิด", format_date(row.get("birth_date", "")) if row.get("birth_date", "") else ""),
            ("น้ำหนัก", (row.get("weight", "") + " กก.") if row.get("weight", "") else ""),
            ("ส่วนสูง", (row.get("height", "") + " ซม.") if row.get("height", "") else ""),
            ("สถานภาพ", row.get("marital_status", "") if row.get("marital_status", "") not in ("", "โสด") else ""),
            ("อาชีพ", row.get("occupation", "") if row.get("occupation", "") not in ("", "นักศึกษา") else ""),
            ("คณะ", row.get("faculty", "")),
            ("ที่อยู่ปัจจุบัน", row.get("current_address", "")),
            ("โทรศัพท์", row.get("phone", "")),
            ("อีเมลผู้รับบริการ", row.get("email", "")),
            ("อีเมลนิสิต", row.get("student_email", "")),
            ("ผู้ติดต่อฉุกเฉิน", row.get("emergency_contact", "")),
            ("โทรศัพท์ผู้ติดต่อฉุกเฉิน", row.get("emergency_phone", "")),
            ("กรุ๊ปเลือด", row.get("blood_group", "")),
            ("สิทธิการรักษา", row.get("treatment_right", "") if row.get("treatment_right", "") not in ("", "ไม่มี") else ""),
            ("ประกันสุขภาพ", row.get("health_insurance", "") if row.get("health_insurance", "") not in ("", "ไม่มี") else ""),
            ("โรคประจำตัว", row.get("chronic_disease", "") if row.get("chronic_disease", "") not in ("", "ไม่มี") else ""),
            ("ประวัติแพ้ยา", row.get("drug_allergy", "") if row.get("drug_allergy", "") not in ("", "ไม่มี") else ""),
            ("ประวัติแพ้อาหาร", row.get("food_allergy", "") if row.get("food_allergy", "") not in ("", "ไม่มี") else ""),
            ("อุณหภูมิ", (row.get("temperature", "") + " °C") if row.get("temperature", "") else ""),
            ("ชีพจร", (row.get("pulse", "") + " /min") if row.get("pulse", "") else ""),
            ("การหายใจ", row.get("respiration", "") if row.get("respiration", "") not in ("", "ปกติ") else ""),
            ("ความดันโลหิต", (row.get("blood_pressure", "") + " mmHg") if row.get("blood_pressure", "") else ""),
            ("SpO₂", (row.get("spo2", "") + " %") if row.get("spo2", "") else ""),
        ]
        for label, value in printable:
            if str(value or "").strip():
                optional_rows.append(tr(label, value))
        updates_html = "".join(optional_rows) or '<tr><td colspan="2">ไม่มีข้อมูลที่ต้องการ Update</td></tr>'
        return f"""
<div class="a4">
  <h1>ใบเวชระเบียนผู้ป่วยนอก (OPD) — คนไข้เก่า</h1>
  <p style="text-align:center">สถานพยาบาลมหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตกำแพงแสน</p>
  <table>
    <tr><th>วันที่/เวลา</th><td>{esc(format_datetime(row.get('created_at_bkk','')))}</td>
        <th>HN</th><td>{esc(row.get('hn',''))}</td>
        <th>Record ID</th><td>{esc(row.get('record_id',''))}</td></tr>
    <tr><th>เลขบัตรประชาชน</th><td colspan="5">{esc(row.get('citizen_id',''))}</td></tr>
  </table>
  <h2>ข้อมูลที่ผู้รับบริการกรอก / ต้องการ Update</h2>
  <table>{updates_html}</table>
  <h2>อาการนำ (ไม่สบาย เป็นอะไร)</h2>
  <div class="complaint-box">{complaint}</div>
  <h2>AI Pre-triage (MTS decision support)</h2>
  <table>
    <tr><th>สีที่แนะนำ</th><td>{esc(row.get('ai_triage_color',''))} {esc(row.get('ai_triage_priority',''))}</td><th>Chart</th><td>{esc(row.get('ai_triage_chart',''))}</td></tr>
    <tr><th>Discriminator</th><td colspan="3">{esc(row.get('ai_triage_discriminator',''))}</td></tr>
  </table>
  <div class="small">AI advisory only — พยาบาล/แพทย์ต้องยืนยันหลังประเมินผู้ป่วยโดยตรง</div>
  <h2>บันทึกเวชระเบียน</h2>
  <div class="complaint-box">{note}</div>
  <div style="height:8mm"></div>
  <table><tr><td style="width:50%">ลายมือชื่อผู้รับบริการ ........................................</td>
      <td>ผู้บันทึก ........................................................</td></tr></table>
</div>"""

    return f"""
<div class="a4">
  <h1>ใบเวชระเบียนผู้ป่วยนอก (OPD)</h1>
  <p style="text-align:center">สถานพยาบาลมหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตกำแพงแสน</p>
  <table>
    <tr><th>วันที่/เวลา</th><td>{esc(format_datetime(row.get('created_at_bkk','')))}</td>
        <th>HN</th><td>{esc(row.get('hn',''))}</td>
        <th>Record ID</th><td>{esc(row.get('record_id',''))}</td></tr>
  </table>
  <h2>ข้อมูลส่วนบุคคล</h2>
  <table>
    <tr><th>ชื่อ</th><td>{esc(row.get('first_name',''))}</td><th>นามสกุล</th><td>{esc(row.get('last_name',''))}</td></tr>
    <tr><th>ชื่อภาษาอังกฤษ</th><td colspan="3">{esc(row.get('english_name',''))}</td></tr>
    <tr><th>บัตรประชาชน</th><td>{esc(row.get('citizen_id',''))}</td><th>รหัสนิสิต</th><td>{esc(row.get('student_id',''))}</td></tr>
    <tr><th>ประเภทผู้รับบริการ</th><td>{esc(row.get('patient_type',''))}</td><th>เพศ</th><td>{esc(row.get('sex',''))}</td></tr>
    <tr><th>เชื้อชาติ</th><td>{esc(row.get('race',''))}</td><th>สัญชาติ</th><td>{esc(row.get('nationality',''))}</td></tr>
    <tr><th>ศาสนา</th><td>{esc(row.get('religion',''))}</td><th>วันเกิด / อายุ</th><td>{esc(format_date(row.get('birth_date','')))} / {esc(row.get('age',''))} ปี</td></tr>
    <tr><th>น้ำหนัก</th><td>{esc(row.get('weight',''))} กก.</td><th>ส่วนสูง</th><td>{esc(row.get('height',''))} ซม.</td></tr>
    <tr><th>สถานภาพ</th><td>{esc(row.get('marital_status',''))}</td><th>อาชีพ</th><td>{esc(row.get('occupation',''))}</td></tr>
    <tr><th>คณะ</th><td colspan="3">{esc(row.get('faculty',''))}</td></tr>
  </table>
  <h2>ข้อมูลติดต่อ</h2>
  <table>
    {tr("ที่อยู่ปัจจุบัน", row.get('current_address',''))}
    <tr><th>โทรศัพท์</th><td>{esc(row.get('phone',''))}</td><th>อีเมลผู้รับบริการ</th><td>{esc(row.get('email',''))}</td></tr>
    <tr><th>อีเมลนิสิต</th><td colspan="3">{esc(row.get('student_email',''))}</td></tr>
    <tr><th>ผู้ติดต่อฉุกเฉิน</th><td>{esc(row.get('emergency_contact',''))}</td><th>โทรศัพท์ผู้ติดต่อฉุกเฉิน</th><td>{esc(row.get('emergency_phone',''))}</td></tr>
  </table>
  <h2>ประวัติสุขภาพ</h2>
  <table>
    <tr><th>กรุ๊ปเลือด</th><td>{esc(row.get('blood_group',''))}</td><th>สิทธิการรักษา</th><td>{esc(row.get('treatment_right',''))}</td></tr>
    <tr><th>ประกันสุขภาพ</th><td>{esc(row.get('health_insurance',''))}</td><th>โรคประจำตัว</th><td>{esc(row.get('chronic_disease',''))}</td></tr>
    {tr("ประวัติแพ้ยา", row.get('drug_allergy',''))}
    {tr("ประวัติแพ้อาหาร", row.get('food_allergy',''))}
  </table>
  <h2>สัญญาณชีพ</h2>
  <table>
    <tr><th>T</th><td>{esc(row.get('temperature',''))} °C</td><th>P</th><td>{esc(row.get('pulse',''))} /min</td><th>R</th><td>{esc(row.get('respiration',''))}</td></tr>
    <tr><th>BP</th><td>{esc(row.get('blood_pressure',''))} mmHg</td><th>SpO₂</th><td>{esc(row.get('spo2',''))} %</td><td colspan="2"></td></tr>
  </table>
  <h2>อาการนำ (ไม่สบาย เป็นอะไร)</h2>
  <div class="complaint-box">{complaint}</div>
  <h2>AI Pre-triage (MTS decision support)</h2>
  <table>
    <tr><th>สีที่แนะนำ</th><td>{esc(row.get('ai_triage_color',''))} {esc(row.get('ai_triage_priority',''))}</td><th>Chart</th><td>{esc(row.get('ai_triage_chart',''))}</td></tr>
    <tr><th>Discriminator</th><td colspan="3">{esc(row.get('ai_triage_discriminator',''))}</td></tr>
  </table>
  <div class="small">AI advisory only — พยาบาล/แพทย์ต้องยืนยันหลังประเมินผู้ป่วยโดยตรง</div>
  <h2>บันทึกเวชระเบียน</h2>
  <div class="complaint-box">{note}</div>
  <div style="height:6mm"></div>
  <table><tr><td style="width:50%">ลายมือชื่อผู้รับบริการ ........................................</td>
      <td>ผู้บันทึก ........................................................</td></tr></table>
</div>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(patient_form(), headers={"X-Robots-Tag": "noindex, nofollow"})


@app.post("/submit", response_class=HTMLResponse)
def submit(
    first_name: str = Form(""),
    last_name: str = Form(""),
    english_name: str = Form(""),
    citizen_id: str = Form(""),
    student_id: str = Form(""),
    patient_type: str = Form(""),
    sex: str = Form("หญิง"),
    race_choice: str = Form("ไทย"),
    race_other: str = Form(""),
    nationality_choice: str = Form("ไทย"),
    nationality_other: str = Form(""),
    religion_choice: str = Form("พุทธ"),
    religion_other: str = Form(""),
    birth_date: str = Form(""),
    weight: str = Form(""),
    height: str = Form(""),
    marital_choice: str = Form("โสด"),
    marital_other: str = Form(""),
    occupation_choice: str = Form("นักศึกษา"),
    occupation_other: str = Form(""),
    faculty: str = Form(""),
    faculty_other: str = Form(""),
    current_address: str = Form(""),
    phone: str = Form(""),
    email: str = Form(""),
    student_email: str = Form(""),
    emergency_contact: str = Form(""),
    emergency_phone: str = Form(""),
    blood_group: str = Form(""),
    treatment_right: str = Form("ไม่มี"),
    health_insurance: str = Form("ไม่มี"),
    chronic_disease: str = Form("ไม่มี"),
    drug_allergy: str = Form("ไม่มี"),
    food_allergy: str = Form("ไม่มี"),
    temperature: str = Form(""),
    pulse: str = Form(""),
    respiration: str = Form("ปกติ"),
    blood_pressure: str = Form(""),
    spo2: str = Form(""),
    chief_complaint: str = Form(""),
):
    cid = digits(citizen_id)
    if len(cid) != 13:
        return HTMLResponse(patient_form("กรุณากรอกเลขบัตรประชาชนให้ครบ 13 หลัก"), status_code=400)
    if patient_type not in {"คนไข้ใหม่", "คนไข้เก่า"}:
        return HTMLResponse(patient_form("กรุณาเลือกคนไข้ใหม่หรือคนไข้เก่า"), status_code=400)
    is_old_patient = patient_type == "คนไข้เก่า"
    if not is_old_patient and (not first_name.strip() or not last_name.strip()):
        return HTMLResponse(patient_form("กรุณากรอกชื่อและนามสกุล"), status_code=400)

    normalized_student_email = student_email.strip().lower()
    is_student = occupation_choice == "นักศึกษา"
    if not is_old_patient and is_student and not student_id.strip():
        return HTMLResponse(patient_form("กรุณากรอกรหัสนิสิต"), status_code=400)
    if not is_old_patient and is_student and (not normalized_student_email or not normalized_student_email.endswith("@ku.th")):
        return HTMLResponse(patient_form("กรุณากรอกอีเมลนิสิตมหาวิทยาลัยที่ลงท้ายด้วย @ku.th"), status_code=400)
    if normalized_student_email and not normalized_student_email.endswith("@ku.th"):
        return HTMLResponse(patient_form("อีเมลนิสิตต้องเป็นอีเมลมหาวิทยาลัย @ku.th"), status_code=400)

    age = calculate_age(birth_date) if birth_date.strip() else ""
    if not is_old_patient and not age:
        return HTMLResponse(patient_form("กรุณาตรวจสอบวันเกิด"), status_code=400)

    faculty_value = faculty.strip()
    if faculty_value == "อื่นๆ":
        if not faculty_other.strip() and not is_old_patient:
            return HTMLResponse(patient_form("เมื่อเลือกคณะ “อื่นๆ” กรุณาระบุชื่อคณะ"), status_code=400)
        faculty_value = f"อื่นๆ: {faculty_other.strip()}"

    timestamp = now_bkk().isoformat()
    record = {field: "" for field in FIELDS}
    record.update({
        "record_id": secrets.token_hex(4).upper(),
        "created_at_bkk": timestamp,
        "updated_at_bkk": timestamp,
        "hn": "",
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "english_name": english_name.strip(),
        "citizen_id": cid,
        "student_id": student_id.strip(),
        "patient_type": patient_type,
        "sex": sex.strip() or "หญิง",
        "race": combine_other(race_choice, race_other),
        "nationality": combine_other(nationality_choice, nationality_other),
        "religion": combine_other(religion_choice, religion_other),
        "birth_date": birth_date,
        "age": age,
        "weight": weight.strip(),
        "height": height.strip(),
        "marital_status": combine_other(marital_choice, marital_other),
        "occupation": combine_other(occupation_choice, occupation_other),
        "faculty": faculty_value,
        "current_address": current_address.strip(),
        "phone": phone.strip(),
        "email": email.strip().lower(),
        "student_email": normalized_student_email,
        "emergency_contact": emergency_contact.strip(),
        "emergency_phone": emergency_phone.strip(),
        "blood_group": blood_group.strip(),
        "treatment_right": treatment_right.strip() or "ไม่มี",
        "health_insurance": health_insurance.strip() or "ไม่มี",
        "chronic_disease": chronic_disease.strip() or "ไม่มี",
        "drug_allergy": drug_allergy.strip() or "ไม่มี",
        "food_allergy": food_allergy.strip() or "ไม่มี",
        "temperature": temperature.strip(),
        "pulse": pulse.strip(),
        "respiration": respiration.strip() or "ปกติ",
        "blood_pressure": blood_pressure.strip(),
        "spo2": spo2.strip(),
        "chief_complaint": chief_complaint.strip(),
        "registration_note": "",
    })

    # AI pre-triage is advisory only. Registration still succeeds if AI is unavailable.
    ai = ai_triage_suggestion(record)
    record.update({
        "ai_triage_color": ai.get("color", "UNDETERMINED"),
        "ai_triage_priority": ai.get("priority", ""),
        "ai_triage_chart": ai.get("chart", ""),
        "ai_triage_discriminator": ai.get("discriminator", ""),
        "ai_triage_rationale": ai.get("rationale", ""),
        "ai_triage_questions": " | ".join(ai.get("questions", []) or []),
        "ai_triage_confidence": ai.get("confidence", "LOW"),
        "ai_triage_model": OPENAI_MODEL if OPENAI_API_KEY else "",
        "ai_triage_status": ai.get("status", ""),
    })

    try:
        append_record(record)
    except requests.HTTPError as error:
        return HTMLResponse(
            patient_form("บันทึกข้อมูลไม่สำเร็จ: " + github_error_text(error)),
            status_code=503 if getattr(error.response, "status_code", 500) in (403, 429) else 500,
        )
    except Exception as error:
        return HTMLResponse(
            patient_form(f"บันทึกข้อมูลไม่สำเร็จ: {type(error).__name__}"),
            status_code=500,
        )

    qr_uri = make_qr_data_uri(record["record_id"])
    display_name = f"{record['first_name']} {record['last_name']}".strip()
    if not display_name:
        display_name = "ผู้รับบริการเดิม"
    queue_time = format_datetime(record["created_at_bkk"])
    qr_html = (
        f'<img class="qr" src="{qr_uri}" alt="QR code {esc(record["record_id"])}">'
        if qr_uri else
        '<div class="alert err">ไม่สามารถสร้าง QR ได้ กรุณาใช้รหัสคิวด้านล่าง</div>'
    )

    body = f"""
<div class="wrap">
  <div class="card no-print">
    <h1>บันทึกข้อมูลเรียบร้อยแล้ว</h1>
    <div class="alert ok">ส่งข้อมูลเข้าสู่ระบบ OPD แล้ว</div>
    <p class="note">กรุณาเก็บบัตรคิวหรือภาพ QR code นี้ไว้แสดงแก่เจ้าหน้าที่เวชระเบียน</p>
    <div class="actions">
      <button type="button" onclick="window.print()">พิมพ์ / บันทึกบัตรคิว</button>
      <a class="btn light" href="/">ลงทะเบียนรายใหม่</a>
    </div>
  </div>

  <div class="queue-ticket">
    <h2>บัตรคิว OPD</h2>
    <div class="queue-meta"><b>สถานพยาบาลมหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตกำแพงแสน</b></div>
    {qr_html}
    <div class="queue-id">{esc(record['record_id'])}</div>
    <div class="queue-meta"><b>ชื่อ:</b> {esc(display_name)}</div>
    <div class="queue-meta"><b>ประเภท:</b> {esc(record['patient_type'])}</div>
    <div class="queue-meta"><b>วันเวลา:</b> {esc(queue_time)} น.</div>
    <div style="margin:3mm 0 2mm;padding:3mm;border:1px solid #d0d5dd;border-radius:9px;text-align:left">
      <div style="text-align:center;margin-bottom:2mm"><b>AI Pre-triage (MTS decision support)</b></div>
      <div style="text-align:center;margin-bottom:2mm">{triage_badge_html(record)}</div>
      <div class="queue-meta"><b>Chart:</b> {esc(record.get('ai_triage_chart','') or '—')}</div>
      <div class="queue-meta"><b>Possible discriminator:</b> {esc(record.get('ai_triage_discriminator','') or '—')}</div>
      <div class="queue-note"><b>เหตุผล:</b> {esc(record.get('ai_triage_rationale','') or '—')}</div>
      <div class="queue-note"><b>ต้องตรวจเพิ่ม:</b> {esc(record.get('ai_triage_questions','') or '—')}</div>
      <div class="queue-note"><b>คำเตือน:</b> เป็นคำแนะนำจาก AI จากข้อมูลที่กรอกและสัญญาณชีพเท่านั้น พยาบาล/แพทย์ต้องประเมินผู้ป่วยโดยตรงและยืนยันหรือเปลี่ยนสี</div>
    </div>
    <div class="queue-note">QR code นี้บรรจุเฉพาะรหัสรายการ กรุณาแสดงบัตรคิวนี้แก่เจ้าหน้าที่เวชระเบียน</div>
  </div>
</div>"""
    return HTMLResponse(html_page("บันทึกสำเร็จ", body))


@app.get("/registry", response_class=HTMLResponse)
def registry(request: Request, date_value: str = ""):
    if not is_registry_authenticated(request):
        body = """
<div class="wrap"><div class="card" style="max-width:520px;margin:40px auto">
<h1>เวชระเบียน OPD</h1>
<p class="note">สำหรับเจ้าหน้าที่</p>
<form method="post" action="/registry/login">
<label class="field">รหัสผ่าน<input type="password" name="password" required autofocus></label>
<div style="height:14px"></div><button>เข้าสู่ระบบ</button>
</form></div></div>"""
        return HTMLResponse(html_page("เวชระเบียน OPD", body))

    chosen = date_value or now_bkk().date().isoformat()
    try:
        rows, _ = read_records()
        rows = [r for r in rows if str(r.get("created_at_bkk", "")).startswith(chosen)]
        rows.sort(key=lambda x: x.get("created_at_bkk", ""))
    except requests.HTTPError as error:
        rows = []
        error_html = f'<div class="alert err">อ่านข้อมูลไม่สำเร็จ: {esc(github_error_text(error))}</div>'
    except Exception as error:
        rows = []
        error_html = f'<div class="alert err">อ่านข้อมูลไม่สำเร็จ: {esc(type(error).__name__)}</div>'
    else:
        error_html = ""

    table_rows = ""
    for r in rows:
        table_rows += f"""<tr>
<td>{esc(format_datetime(r.get('created_at_bkk',''))[-5:])}</td>
<td>{esc(r.get('first_name',''))} {esc(r.get('last_name',''))}</td>
<td>{esc(r.get('patient_type',''))}</td>
<td>{esc(r.get('student_id',''))}</td>
<td>{esc(r.get('hn',''))}</td>
<td>{esc(r.get('chief_complaint','')[:70])}</td>
<td>{triage_badge_html(r)}</td>
<td><a href="/registry/{esc(r.get('record_id',''))}">เปิด</a></td>
</tr>"""
    if not table_rows:
        table_rows = '<tr><td colspan="8" class="small">ไม่พบข้อมูลในวันที่เลือก</td></tr>'

    body = f"""
<div class="wrap">
  <div class="card no-print">
    <div class="actions" style="justify-content:space-between">
      <div><h1>เวชระเบียน OPD</h1><p class="note">รายการผู้รับบริการตามวันบันทึก</p></div>
      <form method="post" action="/registry/logout"><button class="btn light">ออกจากระบบ</button></form>
    </div>
    {error_html}
    <form method="get" action="/registry">
      <label class="field" style="max-width:280px">วันที่<input type="date" name="date_value" value="{esc(chosen)}" onchange="this.form.submit()"></label>
    </form>
    <div style="height:12px"></div>
    <table><thead><tr><th>เวลา</th><th>ชื่อ-นามสกุล</th><th>สถานะ</th><th>รหัสนิสิต</th><th>HN</th><th>อาการนำ</th><th>AI สี</th><th></th></tr></thead>
    <tbody>{table_rows}</tbody></table>
  </div>
</div>"""
    return HTMLResponse(html_page("เวชระเบียน OPD", body))


@app.post("/registry/login")
def registry_login(password: str = Form("")):
    if not PASS_REG or not secrets.compare_digest(password, PASS_REG):
        return HTMLResponse(
            html_page(
                "เข้าสู่ระบบไม่สำเร็จ",
                '<div class="wrap"><div class="card"><div class="alert err">รหัสผ่านไม่ถูกต้อง</div><a class="btn" href="/registry">กลับ</a></div></div>',
            ),
            status_code=401,
        )
    response = RedirectResponse("/registry", status_code=303)
    response.set_cookie(
        "opd_registry_auth",
        auth_token(),
        httponly=True,
        secure=bool(os.getenv("VERCEL")),
        samesite="strict",
        max_age=8 * 60 * 60,
    )
    return response


@app.post("/registry/logout")
def registry_logout():
    response = RedirectResponse("/registry", status_code=303)
    response.delete_cookie("opd_registry_auth")
    return response


@app.get("/registry/{record_id}", response_class=HTMLResponse)
def registry_record(request: Request, record_id: str):
    if not is_registry_authenticated(request):
        return RedirectResponse("/registry", status_code=303)
    try:
        row = find_record(record_id)
    except Exception as error:
        return HTMLResponse(html_page("Error", f'<div class="wrap"><div class="card"><div class="alert err">{esc(type(error).__name__)}</div></div></div>'), status_code=500)
    if not row:
        return HTMLResponse(html_page("ไม่พบข้อมูล", '<div class="wrap"><div class="card"><div class="alert err">ไม่พบ Record ID</div></div></div>'), status_code=404)

    txt = esc(record_text(row))
    body = f"""
<div class="wrap">
  <div class="card no-print">
    <div class="actions">
      <a class="btn light" href="/registry">← กลับรายการ</a>
      <a class="btn secondary" href="/registry/{esc(record_id)}/txt">ดาวน์โหลด Text</a>
      <a class="btn" href="/registry/{esc(record_id)}/print" target="_blank">Preview / พิมพ์ A4</a>
    </div>
  </div>

  <div class="card no-print">
    <h1>{esc(row.get('first_name',''))} {esc(row.get('last_name',''))}</h1>
    <p class="small">Record ID {esc(record_id)} · {esc(format_datetime(row.get('created_at_bkk','')))}</p>
    <form method="post" action="/registry/{esc(record_id)}/update">
      <div class="grid">
        <label class="field">HN<input name="hn" value="{esc(row.get('hn',''))}"></label>
        <label class="field full">หมายเหตุเวชระเบียน<textarea name="registration_note">{esc(row.get('registration_note',''))}</textarea></label>
      </div>
      <div style="height:12px"></div><button>บันทึก HN / หมายเหตุ</button>
    </form>
  </div>

  <div class="card no-print">
    <h2>AI Pre-triage — MTS decision support</h2>
    <p>{triage_badge_html(row)} &nbsp; <b>{esc(row.get('ai_triage_priority',''))}</b></p>
    <p><b>Presentation chart:</b> {esc(row.get('ai_triage_chart','') or '—')}</p>
    <p><b>Possible discriminator:</b> {esc(row.get('ai_triage_discriminator','') or '—')}</p>
    <p><b>เหตุผล:</b> {esc(row.get('ai_triage_rationale','') or '—')}</p>
    <p><b>คำถามที่ควรตรวจเพิ่ม:</b> {esc(row.get('ai_triage_questions','') or '—')}</p>
    <p class="small">Confidence: {esc(row.get('ai_triage_confidence','') or '—')} · Model: {esc(row.get('ai_triage_model','') or '—')} · สถานะ: {esc(row.get('ai_triage_status','') or '—')}</p>
    <div class="alert" style="background:#fffaeb;color:#7a2e0e">AI ไม่ใช่ผู้กำหนดสีสุดท้าย ต้องประเมินผู้ป่วยแบบ face-to-face และยืนยัน/เปลี่ยนสีตาม clinical judgment และ MTS discriminator</div>
  </div>

  <div class="card no-print">
    <h2>Text สำหรับ Copy & Paste ลง HIS</h2>
    <textarea id="hisText" class="mono" readonly>{txt}</textarea>
    <div style="height:10px"></div>
    <button type="button" onclick="copyText()">Copy ทั้งหมด</button>
    <span id="copyStatus" class="small" style="margin-left:10px"></span>
  </div>

  <div class="card">
    <h2 class="no-print">Preview ใบเวชระเบียน A4</h2>
    {a4_content(row)}
  </div>
</div>
<script>
async function copyText() {{
  const box = document.getElementById("hisText");
  try {{
    await navigator.clipboard.writeText(box.value);
    document.getElementById("copyStatus").textContent = "คัดลอกแล้ว";
  }} catch(e) {{
    box.select(); document.execCommand("copy");
    document.getElementById("copyStatus").textContent = "คัดลอกแล้ว";
  }}
}}
</script>"""
    return HTMLResponse(html_page("OPD Record", body))


@app.post("/registry/{record_id}/update")
def registry_update(
    request: Request,
    record_id: str,
    hn: str = Form(""),
    registration_note: str = Form(""),
):
    if not is_registry_authenticated(request):
        return RedirectResponse("/registry", status_code=303)
    try:
        ok = update_record(
            record_id,
            {"hn": hn.strip(), "registration_note": registration_note.strip()},
        )
    except requests.HTTPError as error:
        return HTMLResponse(
            html_page("Error", f'<div class="wrap"><div class="card"><div class="alert err">บันทึกไม่สำเร็จ: {esc(github_error_text(error))}</div></div></div>'),
            status_code=503 if getattr(error.response, "status_code", 500) in (403, 429) else 500,
        )
    except Exception as error:
        return HTMLResponse(
            html_page("Error", f'<div class="wrap"><div class="card"><div class="alert err">บันทึกไม่สำเร็จ: {esc(type(error).__name__)}</div></div></div>'),
            status_code=500,
        )
    if not ok:
        return HTMLResponse(html_page("Error", '<div class="wrap"><div class="card"><div class="alert err">บันทึกไม่สำเร็จ</div></div></div>'), status_code=500)
    return RedirectResponse(f"/registry/{record_id}", status_code=303)


@app.get("/registry/{record_id}/txt")
def download_txt(request: Request, record_id: str):
    if not is_registry_authenticated(request):
        return RedirectResponse("/registry", status_code=303)
    row = find_record(record_id)
    if not row:
        return PlainTextResponse("Not found", status_code=404)
    content = "\ufeff" + record_text(row)
    filename = f"OPD_{record_id}.txt"
    return Response(
        content=content.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/registry/{record_id}/print", response_class=HTMLResponse)
def print_record(request: Request, record_id: str):
    if not is_registry_authenticated(request):
        return RedirectResponse("/registry", status_code=303)
    row = find_record(record_id)
    if not row:
        return HTMLResponse("Not found", status_code=404)
    body = f"""
<div class="wrap">
  <div class="card no-print">
    <div class="actions">
      <button onclick="window.print()">พิมพ์เวชระเบียน + ใบรับรองแพทย์ 2 ใบ</button>
      <a class="btn light" href="/registry/{esc(record_id)}">กลับ</a>
    </div>
    <p class="small">
      พิมพ์ทั้งหมด 2 แผ่น A4: แผ่นแรกเป็นเวชระเบียน OPD และแผ่นที่สองมี
      ใบรับรองแพทย์ขนาดครึ่ง A4 จำนวน 2 ใบ สำหรับตัดแล้วซ้อนกับกระดาษ Carbon copy
    </p>
  </div>
  <div class="print-page">{a4_content(row)}</div>
  <div class="print-page certificate-a4-sheet">
    <div class="medcert-copy-1">{medical_certificate_content(row)}</div>
    <div class="medcert-copy-2">{medical_certificate_content(row)}</div>
  </div>
</div>"""
    return HTMLResponse(html_page(f"OPD {record_id}", body))
