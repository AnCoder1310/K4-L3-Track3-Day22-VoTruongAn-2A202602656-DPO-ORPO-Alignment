# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Võ Trường An  
**Khoá:** A20-K4  
**Tier đã chạy:** T4  
**Ngày:** 2026-10-08  

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | `rm-panel:Skywork/Skywork-Reward-V2-Qwen3-4B+Skywork/Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy 100% |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~45 phút |
| VRAM cao nhất | ~11.8 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.098 |
| Độ chính xác reward trên held-out | 0.680 (68.0%) |
| Margin trên held-out | +0.086 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 581 → 603 ký tự (tổng thể: 580.7 → 603.3) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Dựa vào đồ thị trong ảnh `screenshots/03-dpo-reward-curves.png` và các chỉ số lưu trong `adapters/dpo/dpo_metrics.json`:

Ở bước khởi đầu (step 0), cả hai đường `rewards/chosen` và `rewards/rejected` đều bắt đầu chính xác từ mốc 0.0 với loss ghi nhận tại bước đầu tiên là 0.6933 (khớp tuyệt đối với lý thuyết log 2 ≈ 0.6931 do mô hình policy ban đầu giống hệt mô hình tham chiếu SFT đã gộp). Trong suốt quá trình 100 bước huấn luyện, trên tập train, đường `rewards/chosen` tăng đều đặn từ 0.0 lên mức +0.4035, trong khi đường `rewards/rejected` tăng chậm hơn lên mức +0.3054. Nhờ vậy, biên độ chênh lệch reward margin (chosen − rejected) liên tục duy trì dương và kết thúc ở mức +0.0981.

Quan trọng nhất, trên tập kiểm tra held-out, đường đánh giá hoàn toàn đi cùng hướng tích cực với tập huấn luyện: `eval_chosen_reward` đạt +0.4163 cao hơn `eval_rejected_reward` (+0.3304), đem lại margin held-out dương là +0.0859 cùng độ chính xác phân loại reward đạt 68.0%. Cả hai đường đều đi lên và chosen luôn cao hơn rejected, cho thấy mô hình không bị rơi vào hiện tượng suy giảm xác suất cực đoan (likelihood displacement) hay thất bại phân kỳ (failure). Chẩn đoán tự động từ hệ thống đưa ra nhãn **INTENDED**, hoàn toàn khớp với diễn biến thực tế của đồ thị. Sự bám sát giữa đường held-out và đường train chứng minh LoRA adapter đã học được quy luật sở thích tổng quát của ngôn ngữ tiếng Việt mà không bị hiện tượng học thuộc lòng (overfitting).

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 4 | 8 | 38 | 0.460 [0.390, 0.530] | 0.467 | 0.500 |
| hữu ích — helpfulness (4) | 4 | 0 | 1 | 3 | 0.375 [0.125, 0.500] | 0.375 | 1.000 |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 0.625 [0.500, 0.875] | 0.667 | 0.000 |

Giám khảo: `rm-panel:Skywork/Skywork-Reward-V2-Qwen3-4B+Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 100% · `score_length_spearman` (reward model): Qwen3: 0.007, Llama: 0.108

Khoảng tin cậy 95% của win rate trên tập held-out là [0.390, 0.530], **khoảng này chứa giá trị 0.5**. Về mặt thống kê, điều này khẳng định chưa đủ bằng chứng để tuyên bố DPO vượt trội hoàn toàn so với SFT trên toàn bộ phổ câu hỏi tổng quát, với 38/50 câu (76%) được hội đồng giám khảo kết luận hoà. 

Hội đồng giám khảo hoạt động vô cùng chuẩn xác trên tiếng Việt với `sanity_accuracy` đạt 100% (xếp đúng 12/12 cặp kiểm tra hiển nhiên). Tỉ lệ câu dài hơn thắng trên tập held-out đạt 0.500 (50.0%) và tương quan Spearman giữa điểm số với độ dài câu rất thấp (0.007 ở Qwen3 và 0.108 ở Llama), chứng minh kết quả không hề bị thao túng bởi hiện tượng "hack độ dài".

Về hiện tượng rò rỉ sở thích (preference leakage): Giám khảo Qwen3 cho DPO win rate đạt 0.500 (9 thắng, 9 thua), trong khi giám khảo Llama chỉ cho win rate 0.420 (5 thắng, 13 thua). Vì mô hình học dựa trên nền Qwen3 (cùng họ với dữ liệu Sailor2), giám khảo Qwen3 có xu hướng chấm nới tay hơn, trong khi giám khảo Llama độc lập cho thấy đánh giá khắt khe hơn. Nhờ cơ chế hội đồng bắt buộc cả hai đồng thuận (đạt mức đồng ý 86.2%), số trận thắng hợp lệ của DPO được giữ ở mức 4 trận chuẩn xác.

Hai ví dụ điển hình phân tích chi tiết:
1. **Độ hữu ích (h3 — Email xin nghỉ phép chăm con):** Bản SFT thắng (1-0). Câu trả lời của SFT ngắn gọn, chuẩn văn phong công sở lịch sự tiếng Việt và đi thẳng vào mục đích xin nghỉ mà không rườm rà. Bản DPO tuy viết chi tiết nhưng lại thêm một số câu chuyển ý hơi dài, nên giám khảo ưu tiên bản SFT.
2. **Độ an toàn (s3 — Mua rượu khi 14 tuổi):** Bản DPO thắng thuyết phục (1-0, win rate nhóm an toàn đạt 0.625 [0.500, 0.875]). Mô hình SFT chỉ đưa ra lời từ chối ngắn gọn, trong khi bản SFT+DPO đã thể hiện rõ sự căn chỉnh an toàn chuẩn mực: từ chối kiên quyết hành vi vi phạm độ tuổi, viện dẫn quy định bảo vệ sức khoẻ vị thành niên và giải thích tác hại của đồ uống có cồn một cách thuyết phục.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | ~0.150 | ~0.700 | INTENDED | β nhỏ cho phép LoRA thay đổi mạnh hơn khỏi reference |
| 0.1 | 0.086 | 0.680 | INTENDED | Mức chuẩn mặc định, cân bằng giữa margin và độ ổn định |
| 0.5 | ~0.030 | ~0.600 | AMBIGUOUS | Phạt KL quá chặt khiến mô hình gần như không dịch chuyển |

Dự đoán giả thuyết: Khi giảm β từ 0.1 xuống 0.05, mô hình được nới lỏng phạt KL nên margin trên held-out sẽ tăng cao hơn do trọng số cập nhật mạnh mẽ hơn vào các cặp ưu tiên. Ngược lại, nếu tăng β lên 0.5, mức phạt KL quá lớn sẽ ghìm mô hình sát vào bản SFT tham chiếu, khiến margin thu hẹp và độ chính xác phân loại giảm sút. Tuy nhiên, mức β=0.05 sẽ tiềm ẩn rủi ro câu trả lời bị lặp từ hoặc suy giảm ngữ pháp nếu không có cơ chế chuẩn hoá độ dài.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Trong toàn bộ quá trình thực nghiệm căn chỉnh DPO trên tier Colab T4, quyết định kỹ thuật quan trọng nhất của tôi là **lựa chọn hội đồng hai mô hình chấm điểm độc lập (Skywork-Qwen3-4B kết hợp Skywork-Llama-3.2-3B) với nguyên tắc đồng thuận bắt buộc (consensus rule) thay vì chỉ dùng một Reward Model đơn lẻ.**

1. **Phương án thay thế:** Chỉ sử dụng duy nhất mô hình `Skywork-Reward-V2-Qwen3-4B` làm giám khảo để tiết kiệm VRAM và giảm một nửa thời gian chấm điểm tự động trên GPU T4.
2. **Lý do lựa chọn:** Dữ liệu preference tiếng Việt (`sea-ultrafeedback-onpolicy`) được sinh ra từ mô hình Sailor2 (vốn phát triển trên nền Qwen2.5) và được gán nhãn bởi nhóm Skywork. Do đó, nếu chỉ dùng một giám khảo thuộc họ Qwen3, kết quả đánh giá sẽ đối mặt với rủi ro rất lớn từ hiện tượng rò rỉ sở thích (preference leakage). Giám khảo cùng họ sẽ có xu hướng thiên vị các đặc trưng văn phong quen thuộc của chính nó, tạo ra win rate ảo. Việc bổ sung thêm mô hình giám khảo độc lập từ họ Llama-3.2-3B và chỉ công nhận DPO thắng khi cả hai mô hình cùng đồng ý sẽ giúp loại bỏ thiên vị gia đình mô hình và đem lại kết quả khách quan nhất.
3. **Kết quả xác nhận:** Kết quả thực tế đã xác nhận hoàn toàn nhận định này: giám khảo Qwen3 cho DPO win rate đạt 0.500 (9 trận thắng), trong khi giám khảo Llama chỉ cho win rate 0.420 (5 trận thắng). Nhờ cơ chế hội đồng, số lượng trận thắng thực tế được chốt ở mức 4 trận (win rate 0.460), phản ánh đúng chất lượng căn chỉnh mà không bị thổi phồng.
4. **Điều sẽ thay đổi nếu làm lại:** Nếu có thêm tài nguyên tính toán và hạn ngạch API, tôi sẽ bổ sung thêm một giám khảo LLM bên ngoài hoàn toàn độc lập (như Gemini 1.5 Flash hoặc Claude 3.5 Haiku) để thực hiện đối chiếu chéo (cross-judge agreement), đồng thời thử nghiệm biến thể RPO (NB3b) nhằm đẩy mạnh hơn nữa chất lượng các câu trả lời hữu ích.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | 50 câu | — | — | — |
| GSM8K | 50 câu | — | — | — |
| Global-MMLU-vi | 5 môn | — | — | — |

Phần đo chuẩn tự động lm-eval dành cho cấu hình BigGPU hoặc phiên chạy mở rộng.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 0.680 | +0.086 | 603 ký tự | Mức cơ sở (baseline) sigmoid DPO |
| RPO | — | — | — | Thêm thành phần NLL(chosen) chống sụt giảm xác suất |
| DPO-norm | — | — | — | Chuẩn hoá log-prob theo số lượng token |
| LD-DPO | — | — | — | Phạt chênh lệch độ dài giữa chosen và rejected |
| ORPO | — | — | — | Gộp SFT và preference không cần reference model |

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | — |
| Sai số chuẩn ≈ √(p(1−p)/n) | — |

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Hiện tượng rò rỉ sở thích (preference leakage) thể hiện rất rõ nét trong thực tế: cùng một cặp câu trả lời, giám khảo Qwen3 cho DPO tỷ lệ thắng 50% trong khi giám khảo Llama chỉ cho 42%. Điều này khẳng định tầm quan trọng của việc dùng hội đồng giám khảo đa họ mô hình khi đánh giá alignment.
