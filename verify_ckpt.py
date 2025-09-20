import os, sys, subprocess, json

ckpt = os.environ.get("CKPT_PATH", "/artifacts/model.pt")
out = {"ckpt_path": ckpt, "exists": os.path.exists(ckpt)}
if out["exists"]:
    try:
        sha = subprocess.check_output(
            ["sh", "-lc", f"openssl dgst -sha256 {ckpt} | awk '{'{'}print $2{'}'}'"]
        ).decode().strip()
        out["sha256"] = sha
        out["size_bytes"] = os.path.getsize(ckpt)
    except Exception as e:
        out["error"] = f"sha256_failed: {e}"
print(json.dumps(out, ensure_ascii=False, indent=2))
sys.exit(0 if out["exists"] else 1)
