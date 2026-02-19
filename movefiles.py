import os, shutil

os.makedirs("test-cases", exist_ok=True)

for f in os.listdir("."):
    if f.endswith(".txt"):
        shutil.move(f, "test-cases/")

print("Done!")