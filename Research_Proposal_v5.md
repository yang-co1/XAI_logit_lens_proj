# 다중언어 LLM 잠재 언어 판독의 충실성 검증: Logit Lens는 영어 경유를 과장하는가?

표지훈 · 양동균 · 민지원 · 이가윤
경희대학교 설명및신뢰가능한 AI

---

## Abstract

다국어 LLM이 중간 층에서 영어를 거쳐 "생각한다"는 해석은 주로 logit lens에 근거하지만, logit lens는 최종 층용 언어임베딩 행렬 $W_U$를 기저가 다른 중간 층에 그대로 적용하므로 관찰된 영어 우세 중 얼마나 모델의 실제 연산이고 얼마나 판독 도구의 산물인지 분명하지 않다. 본 연구는 logit lens가 중간 층에서 입력 내용과 무관하게 영어 토큰을 체계적으로 높게 읽어 영어 경유를 과장한다는 가설을 세우고, 이를 판독 도구에 의존하지 않는 행동 기준으로 검증한다.

구체적으로 한국어→프랑스어 단어 번역에서 영어에서만 두 의미가 한 단어로 묶이는 다의어(예: 봄→*spring*→*printemps*/*ressort*)의 간섭을 최종 출력에서 측정하고, logit lens, tuned lens, 평균 성분을 제거한 logit lens가 이 간섭을 얼마나 잘 예측하는지 비교한다. 또한 입력 무관 성분 분해, 영어 방향 제거 개입, 학습 언어 구성과 $W_U$ 공유 여부가 다른 여섯 모델의 비교를 통해 판독 편향의 크기와 원인을 분석한다. 본 연구는 다국어 해석에서 logit lens를 언제 신뢰할 수 있는가에 대한 근거를 제공하는 것을 목표로 한다.

---

## 1. Research Question

> **RQ.** Logit lens로 관찰한 영어는 모델이 영어로 생각한 흔적인가, lens가 만든 왜곡인가?

Wendler 등 [1]은 Français: "fleur" – 中文: "\_\_\_" 프롬프트에서 Llama-2가 정답 "花"를 출력하지만, 중간 층을 logit lens [2]로 디코딩하면 영어 "flower"가 top-1 토큰으로 나타남을 근거로 다국어 LLM이 영어에 가까운 중간 개념 공간을 거친다고 해석하였다. 이후 Bayazit 등 [6]은 logit lens가 tuned lens [3]나 GMM 판독보다 영어를 더 많이 읽음을 보였으나, 그 영어가 모델의 실제 처리인지 lens가 부풀린 것인지는 판정하지 못하였다.

본 연구는 이를 판정하기 위해 최종 출력만으로 측정하는 **영어 다의어 간섭**(Fig. 1)을 기준으로 삼는다. 모델이 "봄"을 번역하면서 *spring*을 거친다면, *spring*의 다른 뜻인 *ressort*(용수철)의 확률도 함께 오를 것이다. logit lens가 모델이 실제로 쓰는 영어를 읽는다면 단어별 lens 영어 점수는 단어별 간섭과 함께 움직여야 한다. 둘의 관련이 약하다면 lens가 보여주는 영어의 상당 부분은 판독 편향으로 해석한다.

**Figure 1. 영어 다의어 간섭.** 영어에서만 *spring*이 '봄'과 '용수철'을 함께 뜻한다.

```
봄 ──(영어 경유?)──▶ spring ──────────▶ printemps (정답)
                       ┆
                       ┆ 간섭
                       ▼
                    ressort (용수철)

여름 ──(대조 단어)──▶ été
```

---

## 2. 관련 연구

**다국어 LLM의 잠재 언어.** Wendler 등 [1]은 번역, 반복, 빈칸 채우기 과제에서 Llama-2의 층별 출력을 logit lens로 추적하였다. 그 결과 초기 층에서는 의미 있는 토큰이 없고, 중간 층에서는 영어 토큰이 우세하며, 최종 층에서야 목표 언어 토큰으로 이동하는 패턴이 일관되게 나타났다. 저자들은 모델이 문자 그대로 영어로 번역한다고 보지는 않았으나, 중간 "개념 공간"이 영어에 가깝다고 주장하였다.

**Logit lens.** logit lens [2]는 층 $\ell$의 hidden state $h_\ell$을 남은 층을 건너뛰고 최종 정규화와 $W_U$로 바로 디코딩한다.

$$p_\ell = \mathrm{softmax}\left(W_U \,\mathrm{LN}_f(h_\ell)\right) \tag{1}$$

학습이 필요 없어 해석 연구의 표준 도구가 되었지만, 중간 층이 최종 층과 같은 기저로 표현되어 있다는 가정을 전제한다. 모델의 학습 목표는 이 가정을 보장하지 않는다.

**Logit lens의 충실성.** 이 가정이 깨진다는 근거가 보고되어 왔다. Belrose 등 [3]은 logit lens 출력이 특정 어휘에 편향되고 층마다 기저가 달라짐을 보이고, 층별 affine 변환을 학습한 tuned lens를 제안하였다. Dumas 등 [4]은 activation patching으로 개념 표현이 언어에 독립적임을 보여, logit lens의 "영어"가 언어 독립적 개념이 $W_U$로 사영된 모습일 수 있음을 시사하였다. Zhong 등 [5]은 잠재 언어가 학습 데이터 분포에 따라 달라짐을 보였다.

---

## 3. 연구 가설

### 3.1 중심 가설

> logit lens는 중간 층에서 입력과 무관하게 영어 토큰을 높게 읽는다. 그 결과 비영어 입력을 비영어로 번역할 때, 모델이 실제로 영어에 의존하는 정도보다 영어 경유를 과장하여 보여준다.

본 연구는 모델이 영어를 전혀 사용하지 않는다고 보지는 않는다. 영어 방향에 대한 개입이 다른 언어의 출력에도 영향을 준다는 근거가 있기 때문이다 [7], [8]. 검증 대상은 모델이 아니라 logit lens의 충실성이다. 검증은 한국어→프랑스어 단어 번역에서, 영어를 포함하지 않는 few-shot 프롬프트로 수행한다.

### 3.2 세부 예측

**예측 1 (입력 무관 성분).** 중간 층 영어 우세의 상당 부분은 모든 입력에 공통인 평균 성분 $\mu_\ell$에서 나온다. 따라서 정답과 무관한 영어 단어도 정답의 영어 번역어처럼 확률이 오르고, 그 폭은 무관한 프랑스어나 한국어 단어보다 크다. 의미 처리가 필요 없는 반복 과제에서도 비슷한 영어 우세가 나타난다. 평균 성분의 영어가 판독 편향인지 실제 상태인지는 이 단계에서 구분할 수 없으므로, 판정은 예측 2에서 수행한다.

**예측 2 (행동 기준과의 불일치).** logit lens의 영어 점수는 영어 다의어 간섭(Fig. 1)을 약하게만 예측하며, 평균 성분을 제거한 판독이 이를 더 잘 예측한다. 간섭은 최종 출력에서만 측정하므로 $W_U$에 의존하지 않는다. 간섭이 언어 독립적 개념 표현에서 생겨나더라도 [4], 두 뜻이 한 단어에 묶이는 것은 영어에서만 일어나므로 이 역시 영어 사용으로 본다.

**예측 3 (인과적 개입).** 중간 층에서 영어 방향을 제거하면 간섭은 줄어들지만, 그 감소율은 logit lens 영어 비중의 감소율보다 작다. 단어별 간섭 감소량도 logit lens 점수와 상관이 약하다.

**예측 4 (모델 간 차이).** 실제 영어 의존과 logit lens 영어 우세 사이의 간극은 영어 중심 모델(Llama-3.2-3B)보다 한·영 이중 언어 모델(EXAONE-3.5-2.4B)에서 크다. 잠재 언어는 학습 데이터에 좌우되지만 [5], $W_U$의 영어 사전 확률은 그대로 남아 있을 것이기 때문이다.

### 3.3 기각 조건

다음 결과가 관찰되면 중심 가설을 기각하고, 해당 조건에서는 logit lens를 신뢰할 수 있다고 결론짓는다.

- 평균 성분을 제거해도 영어 우세가 대부분 유지된다.
- 정답 개념의 영어 번역어만 선택적으로 확률이 오른다.
- logit lens가 간섭을 다른 판독 방법만큼 잘 예측한다.
- 영어 방향 제거 시 logit lens 점수가 높은 단어에서 간섭이 크게 줄어든다.

---

## 4. 데이터셋 및 모델

### 4.1 데이터셋

**단어 과제.** Wendler 등 [1]의 형식을 따라 번역(한국어→프랑스어), 빈칸 채우기, 반복(프랑스어)의 세 과제를 150–300 단어 규모로 구성하고, 프롬프트와 예시는 한국어와 프랑스어로만 작성한다. 영어 번역어가 단일 토큰인 단어만 사용하고, 영어 외래어(예: 마우스)와 프랑스어 정답과 영어 번역어의 첫 토큰이 겹치는 단어(예: *table*)는 제외한다. 두 언어가 라틴 문자를 공유하므로 언어 비중은 영어·프랑스어 사전으로 토큰을 구분해 집계한다. 프랑스어 정답의 토큰 수는 통제 변수로 기록하며, 모델 간 비교에는 공통 문항도 함께 보고한다.

**영어 다의어 간섭 과제.** 대표 영어 번역어가 서로 무관한 두 의미를 갖는 한국어 명사 50–80 개를 구성한다. 한국어 원어나 프랑스어 정답이 영어와 같은 두 의미를 갖는 단어(예: *grue* = 학/기중기)는 제외한다. 대조군으로는 문항마다 다의어가 아니면서 의미 범주와 빈도가 비슷한 단어(예: 봄 ↔ 여름)를 짝짓는다. 단어 빈도와 두 의미의 빈도 비율은 통제 변수로 기록한다.

**일반 텍스트 및 병렬 문장.** 입력 무관 판독 편향은 한국어와 프랑스어 위키백과 문장 각 1,000 개로 측정한다. 언어 방향 계산과 언어 분류 probe에는 FLORES-200 [9]의 한·영·프 병렬 문장을 용도별로 겹치지 않게 나누어 사용한다. tuned lens는 lens 자체가 영어로 치우치지 않도록 한·영·프 위키백과를 같은 비율로 혼합하고 측정용 문장과 겹치지 않는 코퍼스로 학습한다.

### 4.2 모델

| 모델 | 학습 언어 | $W_U$ 공유 |
|---|---|:---:|
| Llama-3.2-3B [10] | 영어 중심 | ✓ |
| Qwen2.5-1.5B [11] | 다국어 | ✓ |
| Gemma-2-2B [12] | 영어위주† | ✓ |
| EXAONE-3.5-2.4B [13] | 한·영 | ✓ |
| Qwen2.5-7B [11] | 다국어 | – |
| Llama-2-7B [14]‡ | 영어 중심 | – |

**Table 1. 실험 모델.** $W_U$ 공유는 입력 임베딩과의 가중치 공유(tie_word_embeddings) 여부. † 다국어 어휘. ‡ 선택.

학습 언어 구성의 효과는 $W_U$를 공유하는 1–3B 모델 네 개를 비교해 분석한다(Table 1). EXAONE-3.5-2.4B는 프랑스어가 주 학습 언어가 아니므로 분석 가능한 문항 수를 함께 보고하며, tuned lens는 1–3B 모델에만 학습한다.

---

## 5. Baseline

**Baseline 1: 개념 무관 단어.** 중간 층 확률 상승이 입력 의미에 반응한 신호인지 lens의 판독 특성인지 구분한다. 동일한 번역 프롬프트에서 정답 개념의 영어 번역어(*spring*)와, 의미가 무관하고 빈도가 유사한 영어·프랑스어·한국어 단어의 중간 층 log-prob을 함께 측정한다. 정답만 선택적으로 상승하면 의미에 반응한 신호이고, 영어 무관 단어도 함께 상승하면 영어 판독 편향으로, 모든 언어의 무관 단어가 함께 상승하면 분포 평탄화로 해석한다.

**Baseline 2: 영어 다의어 간섭.** lens 없이 최종 출력으로 측정하는 간섭 점수(식 (2))를 독립 기준으로 쓴다. 최종 출력 확률만으로 측정하므로 lens에 의존하지 않는 독립 기준이 된다.

**Baseline 3: 비교 lens.** 층별 affine 변환을 학습하여 판독을 보정하는 tuned lens [3]와, 모든 입력에 공통인 성분을 hidden state에서 제거한 뒤 적용하는 logit lens(평균 성분 제거 lens)를 비교 기준으로 사용한다.

---

## 6. 평가 지표

모든 지표는 번역 프롬프트 마지막 토큰의 hidden state $h_\ell$로 계산하며, 모델이 맞힌 문항만 사용한다. 중간 층 구간 $\mathcal{M}$은 상대 깊이 25–75%의 층이다. 단어 확률은 [1]을 따라 그 단어의 첫 토큰 후보들의 확률 합으로 측정한다.

**입력 무관 비율 (예측 1).** 영어 비중 $E$는 lens 분포에서 토큰에 할당된 확률 합을 어휘 내 영어 토큰 비율로 나눈 값이다. $E^c$는 평균 성분을 뺀 $h_\ell - \mu_\ell$로 계산한 영어 비중이며, $\mathcal{M}$에서 평균한 $\rho = 1 - E^c/E$가 클수록 영어 우세가 입력 무관 성분에서 나온다. 개념 무관 단어 기준선에서 중간 층 상승량 $\Delta$의 언어별 평균으로 영어 편향 지수 $B = \bar\Delta_{\mathrm{en}} - (\bar\Delta_{\mathrm{fr}} + \bar\Delta_{\mathrm{ko}})/2$를 구한다.

**간섭 점수.** 오답 의미 단어 $d$(예: *ressort*)에 대해 원어 프롬프트 $x_w$(봄)와 대조 프롬프트 $x_c$(여름)의 최종 출력 확률 차이로 정의한다.

$$I(w) = \log p(d \mid x_w) - \log p(d \mid x_c) \tag{2}$$

**lens 예측력 (예측 2).** 각 lens의 점수 $s_L(w) = \max_{\ell \in \mathcal{M}} \log p_\ell^L(e_w)$는 영어 번역어 $e_w$(예: *spring*)의 중간 층 최대 확률이다. 예측력 $r_L$은 단어 빈도를 통제한 $s_L$과 $I$의 부분 Spearman 상관으로, lens 간 차이는 bootstrap 95% 신뢰구간으로 판정한다.

**개입 효과 (예측 3).** 영어 방향 제거 전후의 간섭 감소율 $\delta_I$와 logit lens 영어 비중 감소율 $\delta_E$를 비교하며, $\delta_E > \delta_I$이면 가설을 지지한다. 유의성은 무작위 방향 100개 제거 대비 $z$-점수로 판정한다.

**모델 간 비교 (예측 4).** 모델마다 판독 간극 $G = \delta_E - \delta_I$와 $\rho$를 공통 문항에서 계산하여 비교한다.

**신뢰도.** 7 절의 프롬프트 변형으로 구한 split-half 신뢰도가 0.7 이상일 때만 단어별 상관을 해석한다.

---

## 7. Stress Test 계획

결과가 단어 집합, 출력 언어, 프롬프트에 한정되지 않는지 확인하기 위해 세 조건에서 같은 분석을 반복하고, 상관 방향과 lens 간 예측력 순위가 유지되면 결론이 견고하다고 본다.

**반대 방향 검사.** 한국어에서만 여러 의미를 갖는 다의어(예: 배 → *ship*/*pear*/*belly*)로 문항을 구성한다. 이때 간섭은 영어가 아닌 한국어 입력의 중의성에서 생겨나므로 영어 lens 점수는 간섭을 예측하지 않아야 한다. 예측한다면 lens 점수가 영어 경유가 아닌 다른 요인(예: 의미적 불확실성)을 반영하는 것으로 본다.

**출력 언어 변경.** 출력 언어를 중국어, 독일어, 스페인어로 바꿔 같은 실험을 반복한다. 특히 중국어는 영어와 문자 체계가 달라, 프랑스어 결과가 라틴 문자 공유의 부산물인지 확인할 수 있다. 결과가 프랑스어에서만 나타난다면 결론의 범위를 해당 언어로 한정한다.

**프롬프트 변형.** few-shot 예시 단어, 언어 표기, 예시 수를 바꾼 프롬프트 4개로 반복 측정하여 split-half 신뢰도를 구한다. 묶음 사이에서 lens 간 예측력 순위가 바뀌면 단어별 상관에 근거한 결론은 보류한다.

---

## 8. 역할 분담

| 이름 | 담당 영역 | 주요 수행 내용 |
|---|---|---|
| 이가윤 | 실험 | 데이터 준비, 본 실험, 실험 결과 공유 |
| 민지원 | 실험 | 데이터 준비, 본 실험, 실험 결과 공유 |
| 양동균 | 보고서 | 최종 보고서 작성, 수정 및 제출 |
| 표지훈 | PPT·발표 | 발표 자료 제작, 발표 |

**Table 2. 팀원별 역할.**

이가윤과 민지원은 실험을 맡고, 양동균은 최종 보고서를, 표지훈은 발표 자료 제작과 발표를 맡는다(Table 2). 실험의 세부 분담과 교차 검토 방식은 추후 정한다.

---

## References

[1] C. Wendler, V. Veselovsky, G. Monea, and R. West, "Do llamas work in English? On the latent language of multilingual transformers," in *Proceedings of the 62nd annual meeting of the association for computational linguistics (ACL)*, 2024.

[2] nostalgebraist, "Interpreting GPT: The logit lens." LessWrong, <https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens>, 2020.

[3] N. Belrose *et al.*, "Eliciting latent predictions from transformers with the tuned lens," *arXiv preprint arXiv:2303.08112*, 2023.

[4] C. Dumas, C. Wendler, V. Veselovsky, G. Monea, and R. West, "Separating tongue from thought: Activation patching reveals language-agnostic concept representations in transformers," in *Proceedings of the 63rd annual meeting of the association for computational linguistics (ACL)*, 2025.

[5] C. Zhong *et al.*, "Beyond English-centric LLMs: What language do multilingual language models think in?" *arXiv preprint arXiv:2408.10811*, 2024.

[6] D. Bayazit, B. AlKhamissi, and A. Bosselut, "Lingua franca or probing artifact? Rethinking latent language in multilingual LLMs," *arXiv preprint arXiv:2609.00155*, 2026.

[7] L. Schut, Y. Gal, and S. Farquhar, "Do multilingual LLMs think in English?" *arXiv preprint arXiv:2502.15603*, 2025.

[8] Z. Wu, X. V. Yu, D. Yogatama, J. Lu, and Y. Kim, "The semantic hub hypothesis: Language models share semantic representations across languages and modalities," in *International conference on learning representations (ICLR)*, 2025.

[9] NLLB Team, "No language left behind: Scaling human-centered machine translation," *arXiv preprint arXiv:2207.04672*, 2022.

[10] A. Grattafiori *et al.*, "The Llama 3 herd of models," *arXiv preprint arXiv:2407.21783*, 2024.

[11] Qwen Team, "Qwen2.5 technical report," *arXiv preprint arXiv:2412.15115*, 2024.

[12] Gemma Team, "Gemma 2: Improving open language models at a practical size," *arXiv preprint arXiv:2408.00118*, 2024.

[13] LG AI Research, "EXAONE 3.5: Series of large language models for real-world use cases," *arXiv preprint arXiv:2412.04862*, 2024.

[14] H. Touvron *et al.*, "Llama 2: Open foundation and fine-tuned chat models," *arXiv preprint arXiv:2307.09288*, 2023.
