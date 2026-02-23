**nvidia-smi (MiB) vs PyTorch (Bytes) — Conversion Comparison**

| Tool | Unit | Converts to GB by | Example | Result |
|---|---|---|---|---|
| nvidia-smi | MiB | ÷ 1024 | 2431 MiB ÷ 1024 | 2.37 GB |
| PyTorch `1e9` | Bytes | ÷ 1,000,000,000 | 2,370,000,000 ÷ 1e9 | 2.37 GB |
| PyTorch `1024**3` | Bytes | ÷ 1,073,741,824 | 2,370,000,000 ÷ 1024³ | 2.21 GB |

---

**Key difference in one line:**

- nvidia-smi goes **MiB → GB** in one step (÷ 1024)
- PyTorch goes **Bytes → GB** in one step (÷ 1e9 or ÷ 1024³)

Both are measuring the same physical memory — just starting from different units and using different divisors, which is why the final numbers are slightly different across tools.

---

**Why T4 shows 14.74 GB in PyTorch but 15.00 GB from nvidia-smi:**

```
nvidia-smi : 15360 MiB ÷ 1024        = 15.00 GB   (decimal)
PyTorch    : 15360 MiB × 1024        = 15,728,640,000 bytes
             15,728,640,000 ÷ 1024³  = 14.74 GB   (binary)
```

Same memory, different math — that 0.26 GB gap is purely a rounding difference between decimal and binary conversion.