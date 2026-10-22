"""ResumeProof extras: evidence check, expert-system rules (forward chaining), skill-gap roadmap,
bullet rewrites, interview questions and blind-screening view. Pure Python, runs offline."""
import re
from collections import Counter
from model import predict_roles

ALIAS = {"machine learning": ["ml"], "javascript": ["js"], "typescript": ["ts"], "postgresql": ["postgres"],
         "kubernetes": ["k8s"], "node": ["node.js", "nodejs"], "react": ["react.js", "reactjs"],
         "scikit-learn": ["sklearn"], "nlp": ["natural language processing"], "mongodb": ["mongo"],
         "power bi": ["powerbi"], "ci/cd": ["cicd", "github actions", "jenkins"], "c++": ["cpp"],
         "deep learning": ["neural network", "cnn", "lstm"], "rest api": ["restful", "rest apis"],
         "networking": ["tcp/ip", "network security"], "statistics": ["statistical"]}
ROLE_CORE = {
    "Data Scientist / ML Engineer": ["python", "machine learning", "pandas", "scikit-learn", "tensorflow", "statistics"],
    "Data Analyst": ["sql", "excel", "tableau", "power bi", "python", "statistics"],
    "Web Developer (Frontend)": ["html", "css", "javascript", "react", "typescript", "git"],
    "Backend Developer": ["python", "sql", "rest api", "docker", "git", "postgresql"],
    "DevOps / Cloud Engineer": ["linux", "docker", "kubernetes", "aws", "ci/cd", "git"],
    "Embedded / IoT Engineer": ["c", "iot", "esp32", "mqtt", "arduino", "rfid"],
    "Mobile App Developer": ["kotlin", "flutter", "firebase", "git", "rest api", "android"],
    "Cybersecurity Analyst": ["linux", "networking", "owasp", "nmap", "python", "encryption"],
    "QA / Test Engineer": ["selenium", "pytest", "postman", "jira", "sql", "git"]}
CLICHES = ["team player", "hard working", "hardworking", "results-oriented", "self-motivated", "detail-oriented",
           "go-getter", "passionate about", "fast learner", "quick learner", "synergy", "think outside the box"]
FIX = {"responsible for": "Managed", "worked on": "Built", "helped with": "Contributed to", "assisted": "Supported",
       "involved in": "Contributed to", "tasked with": "Delivered", "duties included": "Handled"}
QUANT = r"\d+(\.\d+)?\s?(%|x|k|m|\+)|\$\s?\d|\b\d{2,}\b"


def has(text, skill):
    return any(re.search(r"(?<![\w+#])" + re.escape(a) + r"(?![\w+#])", text) for a in [skill] + ALIAS.get(skill, []))


def blind(text):
    t, n = text, 0
    for pat in (r"[\w.+-]+@[\w-]+\.[\w.]+", r"https?://\S+|(?:linkedin|github)\.com\S*", r"\+?\d[\d\s().-]{8,}\d",
                r"(?im)^.*\b(dob|date of birth|gender|marital status|nationality|age)\b.*$"):
        t, c = re.subn(pat, "[hidden]", t, flags=re.I); n += c
    ls = t.splitlines()
    for i, l in enumerate(ls):
        if l.strip():
            ls[i] = "[name hidden]"; n += 1; break
    return {"text": "\n".join(ls), "hidden": n}


RULES = [  # (id, IF-condition, THEN-advice, points, test)
    ("R1", "resume uses first-person pronouns", "Drop I/my/me and start bullets with verbs", 3, lambda f: f["pron"] >= 2),
    ("R2", "resume contains clichés", "Replace clichés with proof (project, number)", 3, lambda f: bool(f["cl"])),
    ("R3", "same opening verb used 3+ times", "Vary your action verbs", 2, lambda f: f["rep"]),
    ("R4", "40%+ of skills have no project/job evidence", "Use each listed skill inside a bullet", 5,
     lambda f: f["nclaim"] and f["nclaim"] / max(1, f["nskill"]) >= .4),
    ("R5", "under 30% of bullets contain numbers", "Add metrics: users, %, time saved, size", 6, lambda f: f["qr"] < .3),
    ("R6", "passive voice found", "Rewrite in active voice", 2, lambda f: f["passive"] >= 2),
    ("R7", "5+ high-weight job keywords missing", "Add the missing keywords where truthful", 6, lambda f: f["nmiss"] >= 5),
    ("R8", "readiness for best-fit role below 50%", "Close the skill gaps in the roadmap", 4, lambda f: f["ready"] < 50)]


def deep_insights(resume, lines, bullets, miss, VERBS):
    low = resume.lower()
    evid = [l.lower() for l in lines if len(l) > 45 and l.count(",") < 3]
    universe = {s for r in ROLE_CORE.values() for s in r} | set(ALIAS)
    skills = sorted(s for s in universe if has(low, s))
    proven = [s for s in skills if any(has(l, s) for l in evid)]
    claimed = [s for s in skills if s not in proven]
    role = predict_roles(resume, 1)[0]["role"]
    core = ROLE_CORE.get(role, [])
    need = [s for s in core if not has(low, s)]
    ready = round(100 * (len(core) - len(need)) / max(1, len(core)))
    tgt = {"role": role, "ready": ready, "need": need, "roadmap": [
        f"{s}: build a small project with it and add one measurable result under Projects." for s in need]}
    openers = Counter((re.findall(r"[a-z]+", re.sub(r"^\s*[-•*▪·]\s*", "", b).lower()) or [""])[0] for b in bullets)
    nb = max(1, len(bullets))
    f = dict(pron=len(re.findall(r"\b(i|my|me)\b", low)), cl=[c for c in CLICHES if c in low],
             rep=any(v >= 3 and k in VERBS for k, v in openers.items()), nclaim=len(claimed), nskill=len(skills),
             qr=sum(bool(re.search(QUANT, b, re.I)) for b in bullets) / nb,
             passive=len(re.findall(r"\b(was|were|been)\s+\w+ed\b", low)), nmiss=len(miss), ready=ready)
    trace = [dict(id=i, cond=c, then=t, gain=g, fired=bool(fn(f))) for i, c, t, g, fn in RULES]
    flags = [x for x in trace if x["fired"]]
    rewrites = []
    for b in bullets:
        t = re.sub(r"^\s*[-•*▪·]\s*", "", b).strip()
        w = (re.findall(r"[a-z]+", t.lower()) or [""])[0]
        issues = (["does not start with a strong verb"] if w not in VERBS else []) + \
                 (["no number or result"] if not re.search(QUANT, t, re.I) else [])
        if not issues or len(rewrites) >= 4: continue
        new = t
        for weak, strong in FIX.items():
            if t.lower().startswith(weak): new = strong + t[len(weak):]; break
        new = new.rstrip(".") + ", improving [metric] by [N%] for [users/team]."
        rewrites.append(dict(old=t[:140], new=new[:200], issue="; ".join(issues)))
    qs = [f"You list {s} but show no project using it. Be ready to explain one real task you did with {s}." for s in claimed[:3]]
    qs += [f"Explain a decision you made using {s}. What alternative did you reject and why?" for s in proven[:3]]
    qs.append(f"Why are you a fit for a {role} role, and which skill from your roadmap would you learn first?")
    return dict(proven=proven, claimed=claimed, target=tgt, rewrites=rewrites, flags=flags, trace=trace,
                gain=sum(x["gain"] for x in flags), questions=qs, blind=blind(resume))
