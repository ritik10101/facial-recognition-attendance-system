import importlib

packages = [
    "flask",
    "mysql.connector",
    "cv2",
    "numpy",
    "pandas",
    "PIL",
    "sklearn",
    "dotenv"
]

print("\n=== Requirements Verification ===\n")

for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"[ OK ] {pkg} installed")
    except Exception:
        print(f"[ MISSING ] {pkg} NOT installed")

print("\n= All Good \n")
