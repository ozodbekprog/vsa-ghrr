# GHRR / HeRR retrofit — qurilish hisoboti

**Sana:** 2026-11-04 · **Mashina:** DGX Spark (`spark-e2aa`), NVIDIA GB10, 121 GB unified, Ubuntu 24.04 aarch64
**Stek:** g++ 13.3, CMake 3.28, Python 3.12, PyTorch 2.9.1+cu130, transformers 4.46.3

---

## 1. Bir qarashda

| Nima | Holat |
|---|---|
| C++ VSA yadrosi (FHRR + GHRR + HeRR) | ✅ qurildi, 39 test o'tdi |
| Sig'im / shovqin benchmarklari | ✅ o'lchandi (pastda) |
| pybind11 bog'lanishi | ✅ `vsacore_py` ishlaydi |
| PyTorch GHRR attention | ✅ 11 test o'tdi, **asl modelni aynan takrorlaydi** |
| 1B modelni retrofit | ✅ bajarildi (Qwen2.5-0.5B — pastdagi izohga qarang) |
| LoRA fine-tune + perplexity | ✅ 4 ta tajriba o'tkazildi |
| GPU | ✅ GB10'da matmul va o'qitish ishladi |

**Eng muhim natija:** GHRR attention — bu matematik jihatdan **aynan** RoPE attention.
`diagonality=1.0` da o'zgartirilgan modelning logitlari asl modelnikidan `1.5e-7`
farq qiladi, WikiText-2 perplexity esa `15.7446 → 15.7453` (`Δ = +0.0008`).
Ya'ni attentionni almashtirish **hech narsani buzmaydi** — bu qurilishning eng
qiyin va eng muhim tekshiruvi.

---

## 2. Nima qurildi

```
vsa-ghrr/
├── include/vsa/{common,fhrr,ghrr,herr}.hpp   C++ yadro (header-only)
├── src/bench.cpp                              capacity / noise / algebra
├── tests/test_vsa.cpp                         39 algebra testi
├── python/bindings.cpp                        pybind11 moduli
├── python/gpuvsa/attention.py                 GHRRPositionBinder + GHRRAttention
├── python/gpuvsa/retrofit.py                  perplexity, LoRA, hisobot
├── scripts/run_pipeline.py                    to'liq tajriba
├── tests/test_attention.py                    11 ta tekshiruv (CPU/GPU)
└── out/*.json                                 o'lchangan natijalar
```

### Matematik yadro

- **FHRR** — baseline: birlik modulli kompleks vektorlar, bind = elementwise ko'paytirish.
- **GHRR** — `D` ta `m×m` unitar matritsa. Bind = elementwise matritsa ko'paytmasi,
  unbind = konjugat transpozitsiya. `diagonality` parametri qaysi qismi diagonal
  (kommutativ) ekanini boshqaradi.
- **HeRR** — Laurent polinomlari: ko'rsatkichlar `ℤ` da, bind = polinom ko'paytmasi,
  dekoder = matched filter + **mean-and-peel** (eng kuchli juftlikni topib, ayirib tashlash).

Ishlab chiqishda topilgan ikkita muhim xato (ikkalasi ham tuzatildi va testlar bilan qoplandi):

1. **GHRR unbind tomoni.** `bind(a,b) = a·b` bo'lganda `(a·b)·a† = a·b·a† ≠ b`.
   To'g'ri dekoder — **chapdan** unbind: `a†·(a·b) = b`. Bu tuzatilmaguncha
   `diagonality=0` da sig'im 0.26 edi; tuzatilgandan keyin 1.00 bo'ldi.
2. **Tasodifiy unitar matritsalarning nolga teng bo'lmagan o'rtachasi.** Diagonalni
   real qilish (faza tuzatish) mustaqil vektorlar orasida soxta `0.28` o'xshashlik
   berardi — Haar taqsimotidan foydalanib tuzatildi.

---

## 3. Sig'im natijalari (o'lchangan)

### FHRR — `M` juftlikdan rolni to'g'ri dekodlash aniqligi

| D | M=8 | 16 | 32 | 64 | 96 | 128 | 192 |
|---|---|---|---|---|---|---|---|
| 64 | 1.00 | 0.86 | 0.51 | 0.17 | 0.10 | 0.06 | 0.03 |
| 128 | 1.00 | 0.98 | 0.76 | 0.40 | 0.20 | 0.15 | 0.07 |
| 256 | 1.00 | 1.00 | 0.93 | 0.67 | 0.43 | 0.26 | 0.17 |
| 512 | 1.00 | 1.00 | 1.00 | 0.93 | 0.77 | 0.60 | 0.34 |
| 1024 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.92 | 0.69 |

### GHRR — teng koeffitsiyent sonda FHRR bilan bir xil

| konfiguratsiya | koeff. | M=8 | 16 | 32 | 64 | 96 |
|---|---|---|---|---|---|---|
| D=64, m=2, diag=0 | 256 | 1.00 | 1.00 | 0.94 | 0.66 | 0.43 |
| D=128, m=2, diag=0 | 512 | 1.00 | 1.00 | 1.00 | 0.94 | 0.78 |
| D=256, m=2, diag=0 | 1024 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 |
| D=128, m=3, diag=0 | 1152 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 |

> FHRR D=512, M=64 → 0.93; GHRR 512 koeff., M=64 → 0.94. Farq yo'q.
> GHRR'ning haqiqiy afzalligi — **sig'im emas, tartib-sezgirlik**: `diag=0` da
> `sim(a⊙b, b⊙a) = 0.48` (m=3), ya'ni bog'lanish kommutativ emas va tartib saqlanadi.

### HeRR — sparse filler bilan

| dekoder | k | M=8 | 16 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|
| one-shot | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| peel | 8 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | 0.97 |
| one-shot | 32 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| peel | 32 | 0.97 | 0.96 | 0.92 | 0.78 | 0.68 | 0.47 |
| dense nazorat | 128 | 0.97 | 0.82 | 0.61 | 0.41 | 0.24 | 0.07 |

### Shovqin (FHRR, D=128, M=16)

| SNR (dB) | −6 | −3 | 0 | 3 | 6 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|---|---|
| aniqlik | 0.50 | 0.65 | 0.87 | 0.92 | 0.97 | 0.98 | 0.97 | 0.97 |

---

## 4. GHRR attention — nima uchun aynan ishlaydi

Head vektori `D = head_dim/m` ta blokka bo'linadi. Har bir blok pozitsiya bilan
**unitar matritsa** orqali bog'lanadi:

```
E_b(pos) = expm(pos · G_b),      G_b — real skew-simmetrik
q'_b = E_b(pos_q) · q_b,         k'_b = E_b(pos_k) · k_b
```

`E(a)E(b) = E(a+b)` bo'lgani uchun skor faqat **nisbiy** pozitsiyaga bog'liq:
`⟨q'_i, k'_j⟩ = ⟨q_i, E(pos_j − pos_i) k_j⟩`.

Llama/Qwen RoPE juftlari `(i, i + head_dim/2)` bo'ladi — shuning uchun head vektori
avval shu tartibga **permutatsiya** qilinadi, keyin `m=2` bloklarda
`G_b = θ_b · [[0,−1],[1,0]]` olinadi. Bu **aynan RoPE**.

`diagonality < 1` bo'lsa, bloklarning bir qismiga nolga teng bo'lmagan, o'rganiladigan
`skew-simmetrik` qo'shimcha generator qo'shiladi (`extra`). U noldan boshlanadi —
shuning uchun model **boshida aynan RoPE**, keyin o'rganishi mumkin.

### Tekshiruvlar (`tests/test_attention.py`, 11/11)

| Test | Natija |
|---|---|
| `diag=1.0` asl modelni takrorlaydi | max farq **1.5e-7** |
| `diag=0.5` ham boshida aynan RoPE | max farq **1.5e-7** |
| `E(pos)` ortogonal | 3.6e-6 |
| Guruh qonuni `E(3)E(4)=E(7)` | 1.3e-6 |
| Kelajak tokeni orqaga oqmaydi | 0.0 |
| Generator gradient oladi | 4.9e-4 |
| RoPE generatorlari muzlatilgan buffer | ✅ |

---

## 5. Retrofit va LoRA natijalari

**Model:** `Qwen2.5-0.5B` (24 qatlam, hidden 896, 14 head, head_dim 64, GQA 2 kv-head,
`rope_theta=1e6`, q/k/v da bias bor).
**O'lchov:** WikiText-2 test, 40 512 token, oyna 1024 / qadam 512.
**LoRA:** r=16, alpha=32, q/k/v/o, 300 qadam × 512 token (WikiText-2 train).

| Tajriba | Retrofitdan keyin | LoRA'dan keyin | Δ baseline |
|---|---|---|---|
| Baseline (o'zgartirilmagan) | — | **11.2681** | −4.48 |
| GHRR, `diag=1.0` (muzlatilgan RoPE) | 15.7453 | **11.2539** | −4.49 |
| GHRR, `diag=0.5`, `lr_gen=1e-3` | 15.7453 | **11.3569** | −4.39 |
| GHRR, `diag=0.5`, `lr_gen=1e-2` | 15.7453 | **11.7215** | −4.02 |

Baseline perplexity: **15.7446**.

### Bu raqamlar nima deydi

1. **Retrofit yo'qotishsiz** — 15.7446 → 15.7453 (+0.0008, fp16/bf16 shovqini).
2. **LoRA ishlaydi** — 300 qadamda −28.5 %. Lekin bu GHRR'ning emasi: asl
   model ham xuddi shunday yaxshilanadi (11.2681). Ya'ni yaxshilanish
   WikiText-2 domenga moslashuvdan keladi, arxitekturadan emas.
3. **O'rganiladigan GHRR bloklari bu byudjetda foyda bermadi** —
   `11.2539` (muzlatilgan) vs `11.3569`/`11.7215` (o'rganiladigan). Past
   `lr_gen` bilan farq kichrayadi, ya'ni natija shovqin chegarasida va
   qisqa o'qitishda **neytral**.

4. **Tezlik:** retrofitdan keyin eval 3.84 s → 5.65 s (40.5k token). Sabab —
   bizning attention `eager` (SD PA emas), birlashtirilgan kernel yo'q. Bu
   optimallashtirish mumkin, lekin hozircha 1.5× sekinlik bor.

---

## 6. Halol xulosa: rejadagi da'volar vs o'lchov

| Rejadagi da'vo | O'lchangan natija |
|---|---|
| "HeRR 128 kompleks sonda 96 juftlikni 1.00 bilan qaytaradi" | **128 koeffitsiyentda emas.** 96 juftlik 1.00 uchun xotira **4141 koeffitsiyent** (32×) kerak bo'ldi. 128 da FHRR 0.20, HeRR ham chegaralangan. |
| "FHRR o'sha joyda 0.13" | FHRR D=128, M=96 → **0.20** (bizning o'lchovda). Sifat jihatdan to'g'ri, soni boshqacha. |
| "GHRR sig'imi yuqori" | Teng koeffitsiyent sonda **FHRR bilan bir xil**. Afzallik — kommutativ emaslik (tartib). |
| "WikiText-2 da 5.47 %, PTB da 2.75 % yaxshilanish" | **Tasdiqlanmadi.** 300 qadamlik LoRA'da GHRR bloklari neytral/minus. Uzoqroq o'qitish kerak. |
| "C++ kutubxonasi tayyor emas" | To'g'ri edi — **biz yozdik**, 39 test bilan. |
| "1B model" | **Qwen2.5-0.5B** ishlatildi: 1.1B TinyLlama 4.2 GB (fp32) va Spark WAN ~0.4 MB/s edi. Quvur istalgan Llama/Qwen2 modelida ishlaydi — `--model` ni o'zgartirish kifoya. |
| "gpt_hrr.py / HoloVec mavjud" | Bunday tayyor paketlar topilmadi; hammasi noldan yozildi. |

**Muhandislik xulosasi:** attentionni GHRR bilan almashtirish — bu ishlaydigan,
tekshiriladigan va yo'qotishsiz operatsiya. Lekin u o'z-o'zidan modelni
yaxshilamaydi: GHRR bu yerda **yangi pozitsion bog'lash algebrasi**, mo''jiza emas.
Foyda olish uchun (a) uzoqroq o'qitish, (b) `diagonality` va `m` ni sozlash,
(c) ehtimol V-qismini ham bog'lash kerak.

---

## 7. Qanday takrorlash

```bash
# yadro
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/vsa-tests
./build/vsa-bench capacity --trials=8 > out/capacity.jsonl
./build/vsa-bench herr --trials=5     > out/herr.jsonl
./build/vsa-bench noise               > out/noise.jsonl

# attention testlari
python tests/test_attention.py

# retrofit
python scripts/run_pipeline.py --model Qwen/Qwen2.5-0.5B \
    --diagonality 0.5 --steps 300 --max-tokens 40000 --out out/run.json
```

1.5B ga o'tish: `--model Qwen/Qwen2.5-1.5B`. Kod o'zgarmaydi.

---

## 8. Keyingi qadamlar (tavsiya)

1. **1.5B / 3B modelda takrorlash** — 0.5B juda kichik, GHRR bloklariga
   sig'adigan signallar kam.
2. **`m=4` va `diagonality=0.25`** variantlarini sinash — hozir faqat `m=2`.
3. **V rolining bog'lanishi** — hozir faqat Q va K bog'lanadi.
4. **Fused kernel** — `eager` attention 1.5× sekinlik beradi; SDPA bilan
   moslashtirish yoki `torch.compile` (triton kerak).
5. **Uzoqroq o'qitish** (3–5k qadam) — GHRR bloklari foyda berishi uchun.
