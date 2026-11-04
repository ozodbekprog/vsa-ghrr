# HGA-1 — birinchi tajriba hisoboti

**Sana:** 2026-11-04 · **Mashina:** DGX Spark (`spark-e2aa`), NVIDIA GB10, Ubuntu 24.04 aarch64
**Stek:** Python 3.12, PyTorch 2.9.1+cu130, transformers 4.46.3, g++ 13.3
**Kod:** Spark `~/vsa-ghrr/python/hga/`, Mac nusxasi `~/Documents/dgx-ghrr/python/hga/`

---

## 1. Bir qarashda

| | HGA-1 | Baseline (BPE) |
|---|---|---|
| Kirish | harflar → FHRR → proyeksiya | 16k BPE token |
| Attention | GHRR (m=4, blok-unitar pozitsiya) | softmax |
| Parametr | **17.50M** | 17.18M |
| Qadam | 3000 | **4270** |
| O'qitish vaqti | 1073 s | 1088 s |
| Ko'rilgan matn | 298.7 MB | 299.5 MB |
| **Test BPB** | **0.8487** | 1.2605 |
| Test ppl | 37.24 (so'z) | 42.77 (token) |

**Asosiy natija: teng hisoblash va teng matn byudjetida HGA-1 BPB bo'yicha
32.7 % yaxshiroq** (`1 − 0.8487/1.2605`).

Nega baseline 4270 qadam? Bir qadamda ikkala model ham 16 384 birlik ko'radi,
lekin so'z o'rtacha 6.08 bayt, BPE token 4.28 bayt. Teng matn uchun baseline
`3000 × 6.08/4.28 ≈ 4270` qadam olishi kerak. Ikkala yugurish ham ~1080 s
davom etdi, ya'ni **teng devor-soat**.

---

## 2. Nima qurildi (4 parallel agent + integratsiya)

| Komponent | Fayl | Tekshiruv |
|---|---|---|
| C++ `CharEncoder` (FHRR random indexing, D=512) | `include/vsa/char_encoder.hpp` | 19 test: self-cos 1.0, `cos(olma,amlo) = −0.032` |
| `HGA1Model` + `GHRRBlock` | `python/hga/model_hga.py` | 17.50M param, causal, grad, 3-step overfit 6.32→4.84 |
| `BaselineLM` + BPE | `python/hga/model_baseline.py`, `bpe.py` | 17.18M param, tied embedding, causal |
| `train.py` / `eval.py` / BPB | `python/hga/*.py` | smoke, resume, checkpoint |
| Ma'lumot quvuri | `python/hga/data.py` | 359 MB matn, 59.1M so'z, 84.1M BPE token |

### Arxitektura bo'yicha qaror

Spetsifikatsiyada `D_ghrr = 96 = heads soni`, lekin `n_heads = 6` — bu ikkisi mos
kelmaydi va 96 blok × 4×4 `d_model=384` ga sig'maydi. O'lchovli izchil variant
ishlatildi:

```
d_model=384, n_heads=6, head_dim=64, m=4  ->  D = head_dim/m = 16
```

`diagonality=1.0`, generatorlar muzlatilgan (avvalgi tajribada o'rganiladigan
qo'shimchalar qisqa byudjetda foyda bermagani o'lchangan). Ya'ni GHRR bloklari —
**aniq nisbiy pozitsiya kodlashuvi**, RoPE'ning 4×4 umumlashmasi.

---

## 3. Ma'lumot

Manba: `wikimedia/wikipedia` 2023-11-01, o'zbek bo'limi (210 MB parquet).

| Split | Bayt | So'z | BPE token |
|---|---|---|---|
| train | 359.4 MB | 59.15M | 84.08M |
| val | 3.85 MB | 0.63M | 0.90M |
| test | 3.96 MB | 0.64M | 0.92M |

So'z lug'ati 16 000, qamrov **85.03 %** (15 % `<unk>`). O'rtacha so'z 6.08 bayt,
BPE token 4.28 bayt.

---

## 4. `<unk>` ta'siri (asosiy e'tirozga javob)

So'z darajasidagi model `<unk>` ni oson bashorat qiladi, bu BPB'ni sun'iy
yaxshilashi mumkin. O'lchandi:

| Ko'rsatkich | Qiymat |
|---|---|
| `<unk>` ulushi | 15.0 % token |
| `<unk>` ning CE'dagi ulushi | **5.9 %** |
| CE / `<unk>` token | 1.41 nat |
| CE / oddiy token | 4.26 nat |
| CE / so'z (hammasi) | 3.618 nat |

15 % token bo'lsa-da, ular CE'ning faqat 5.9 % ini tashkil qiladi. Agar `<unk>`
nishonlarini butunlay chiqarib tashlasak, `CE/non-unk = 4.01 nat` va BPB
taxminan **0.97** bo'ladi — baribir baseline'dan ~23 % yaxshiroq. Ya'ni afzallik
`<unk>` hisobiga emas.

---

## 5. Semantik test (`olma → meva`)

Chiqish embeddinglari (`lm_head` qatorlari) bo'yicha kosinus:

| Juftlik | Kosinus |
|---|---|
| olma ↔ meva | **0.5902** |
| olma ↔ daraxt | 0.5396 |
| olma ↔ stol | 0.5443 |

- `cos(olma, meva) > cos(olma, stol)` ✅
- `cos(olma, daraxt) > cos(olma, stol)` ❌ (0.5396 vs 0.5443 — 0.005 farq)

Model "olma — meva" yaqinligini o'rgangan, "olma — daraxt" ajratilmadi.
3000 qadam va 16k lug'at uchun kutilgan holat.

---

## 6. Tezlik va xotira

| | HGA-1 | Baseline |
|---|---|---|
| O'qitish | 45 804 so'z/s = **278 kB/s** | 64 318 token/s = **275 kB/s** |
| Eval | 156 521 so'z/s | 169 648 token/s |
| Test to'plami | 4.1 s | 5.4 s |
| Checkpoint | 263 MB | 197 MB |

Bayt/sekundda ikkala model amalda bir xil (278 vs 275 kB/s) — GHRR matritsa
amallari bu o'lchamda deyarli hech narsa qimmatlashtirmaydi (chunki m=4 va
`D=16`, ya'ni bosh vektori 64 o'lchamda qoladi).

---

## 7. Halol xulosa

**Ishladi:**
- To'liq quvur: parquet → char encoder → FHRR → GHRR transformer → keyingi so'z → BPB.
- **Teng hisoblashda BPB 0.8487 vs 1.2605 — 32.7 % afzallik.**
- `<unk>` ta'siri o'lchandi va afzallikning asosiy sababi emasligi ko'rsatildi.
- Semantik testning yarmi o'tdi.

**Cheklovlar:**
- 15 % `<unk>` — 16k lug'at o'zbek morfologiyasi uchun kichik; 32k kerak.
- Faqat 0.83 epoch; ikkala model ham to'yilmagan (train loss 3.76 / 3.72).
- Bitta seed — 32.7 % katta farq, lekin 3–5 seed bilan tasdiqlash kerak.
- Spetsifikatsiyadagi 50 000 qadam (~13 epoch) bajarilmadi: 3000/4270.
  To'liq rejim taxminan 8–10 soat.
- **Ablatsiya yo'q:** bu natija GHRR'ning o'zidanmi yoki **so'z darajasidagi
  kirish + FHRR kompozitsiyasidanmi** — ajratilmagan. To'g'ri nazorat: bir xil
  so'z kirishli, lekin oddiy RoPE attentionli model.

**Muhim tuzatish:** dastlab baseline 3000 qadamda o'qitilgan va BPB xato
hisoblangan (baytlar so'z birligida masshtablangan). Ikkalasi ham tuzatildi:
baseline 4270 qadamgacha o'qitildi va `eval.py` BPE birligidan foydalanadi.

---

## 8. Qanday takrorlash

```bash
python python/hga/data.py --parquet data/uzwiki.parquet --out data

python python/hga/train.py --model hga --data data --out runs/hga_3000.jsonl \
  --ckpt runs/hga_3000.pt --steps 3000 --batch 32 --seq 512 --lr 3e-4 --warmup 200 \
  --device cuda --dtype bf16
python python/hga/eval.py --model hga --ckpt runs/hga_3000.pt --data data --split test \
  --out runs/eval_hga.json

python python/hga/train.py --model baseline --data data --out runs/base_4270.jsonl \
  --ckpt runs/base_4270.pt --steps 4270 --batch 32 --seq 512 --lr 3e-4 --warmup 200 \
  --device cuda --dtype bf16
python python/hga/eval.py --model baseline --ckpt runs/base_4270.pt --data data \
  --split test --out runs/eval_base_4270.json
```

---

## 9. Keyingi qadamlar (tavsiya)

1. **Ablatsiya (eng muhim):** HGA-1 kirishi + oddiy RoPE attention. Bu GHRR
   hissasini ajratadi.
2. **32k lug'at** — `<unk>` 15 % → ~5 %.
3. **3–5 seed** — 32.7 % farqning barqarorligini tekshirish.
4. **To'liq rejim** (20–50k qadam).
5. **Char-level baseline** — FHRR kompozitsiyasining o'zini baholash.
