# Does Spanish maths RLVR make an aligned model more compliant in English?

*Draft for LessWrong, 2026-09-25. Numbers are the 250-step results; the per-checkpoint pass is in progress.*

Yong & Bach ([Self-Jailbreaking](https://arxiv.org/abs/2510.20956)) show that benign maths and code reasoning training raises StrongREJECT attack success on aligned models from under 5% to 60–95%; the models still classify the prompts as unsafe but reason their way to compliance. Dobler et al. ([GRPO Beyond English](https://arxiv.org/abs/2608.13698)) show that non-English GRPO transfers across languages but causes model- and language-specific regressions. Neither asks whether RLVR in one language erodes a safety behaviour in another. So I ran an initial experiment.

**Setup.** Qwen2.5-7B-Instruct, LoRA-GRPO for 250 steps on 1,643 mAceReason-Math problems in Spanish, with the same problems in English as the control. Binary Math-Verify reward, one seed per arm, no thinking mode, so the response format is unchanged. Safety is read only after a capability gate (avg@8 gain with a CI excluding zero), on all 313 StrongREJECT prompts, three responses each, judged by GPT-5 with the AISI prompt, Yong & Bach's setup.

**Results.** The Spanish arm gained +5.0 pp in maths ([+1.8, +8.2]) and raised the mean harmfulness score by +0.012 ([+0.005, +0.019]); ASR moved from 9.9% to 12.4%. The English arm gained +2.2 pp ([−0.9, +5.2]), failing the gate, with a null safety change (+0.002, [−0.005, +0.009]). The finals differ by +0.010 ([+0.003, +0.018]). The effect is real but small, about 1.2× the minimum detectable effect and an order of magnitude below self-jailbreaking. Expected: on-policy LoRA training that keeps the response format is far lighter than installing a reasoning mode. Two caveats: the Spanish arm changed more, so dose and language are confounded, and 13% of its Spanish responses drifted into English.

**Next.** I am scoring every 25-step checkpoint of both arms for dose–response curves, then regressing safety change on capability gain with a language term. If Spanish sits above English at equal gain, language is doing the work.

**Extensions worth running:**

- *Other domains.* Maths has a verifiable reward and a parallel multilingual dataset. Code is next; non-verifiable domains need a judge as reward, adding a confound.
- *SFT vs RLVR.* Yong & Bach saw the effect under both. An SFT arm on the same problems' solutions would show whether on-policy RL is the gentler path.
- *Broader safety evals.* Compliance is one behaviour. Sandbagging, sycophancy and over-refusal would separate general drift from targeted erosion.
- *Other models.* Qwen2.5 has intact refusals and no prior RLVR. Gemma-3 and Llama-3.1 would test whether the effect is model-specific, as GRPO Beyond English's regressions were.

Code, data and reports: [github.com/rharish96/rlvr-crosslingual-safety](https://github.com/rharish96/rlvr-crosslingual-safety)
