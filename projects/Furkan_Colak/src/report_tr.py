"""Turkish strings for the tables written by report_ext and report_err (merged into report.TR).

The wording follows paper/tr/main.tex: yeniden ifade, prompt, sıfır örnekli, özel ad, firma.
"""

TR_EXT = {
    # external.tex
    "Human-written CTI by publisher. All excerpts are from documents released on or before 30 November 2022.":
        "Yayıncıya göre insan yazımı CTI. Bütün parçalar 30 Kasım 2022 veya öncesinde yayımlanmış belgelerden alınmıştır.",
    "Publisher & Sources & Excerpts & Released & Median words & IQR words & Role":
        "Yayıncı & Kaynak & Parça & Yayın tarihi & Medyan kelime & IQR kelime & Rol",
    "Train, val., test": "Eğitim, doğrulama, test",
    "Independent test": "Bağımsız test",
    # publishers.tex
    "Models trained on CISA + ABC, threshold 0.5. FPR on all human excerpts of the publisher; recall, F1 and AUC "
    "on human excerpts paired with their LLM versions from A, B and C, averaged. FPR and F1 are mean $\\pm$ std over "
    "three seeds; the interval resamples sources. The last column adds the other external publisher to training.":
        "CISA + ABC ile eğitilen modeller, eşik 0.5. FPR yayıncının bütün insan parçaları üzerinde; duyarlılık, F1 ve "
        "AUC insan parçaları ile bunların A, B ve C sürümleri üzerinde, ortalama. FPR ve F1 üç seed üzerinden ortalama "
        "$\\pm$ std; aralık kaynakları yeniden örnekler. Son sütunda diğer dış yayıncı eğitime eklenmiştir.",
    "Publisher & Detector & FPR & 95\\% CI & Recall & F1 & ROC-AUC & FPR, other publisher in training":
        "Yayıncı & Dedektör & FPR & \\%95 GA & Duyarlılık & F1 & ROC-AUC & FPR, diğer yayıncı eğitimde",
    # calibration.tex
    "Probability quality on test data, training on one of A, B, C (mean $\\pm$ std over three seeds). "
    "Calibrators are fitted on validation data only.":
        "Test verisinde olasılık kalitesi, A, B veya C'den biriyle eğitim (üç seed üzerinden ortalama $\\pm$ std). "
        "Kalibratörler yalnızca doğrulama verisiyle kurulur.",
    "Calibrator": "Kalibratör",
    "Temperature": "Sıcaklık",
    "Isotonic": "İzotonik",
    " & Raw & ": " & Ham & ",
    # thresholds.tex
    "Effect of the decision rule, training on one of A, B, C. Every threshold is set on validation data. "
    "Pseudo-unseen uses validation texts of local generators that are neither in training nor the test generator. "
    "Human 1\\% and 5\\% use validation human scores alone.":
        "Karar kuralının etkisi, A, B veya C'den biriyle eğitim. Bütün eşikler doğrulama verisinde belirlenir. Sözde "
        "görülmemiş kural, ne eğitimde ne de testte bulunan yerel üreticilerin doğrulama metinlerini kullanır. İnsan "
        "\\%1 ve \\%5 kuralları yalnızca doğrulama insan skorlarını kullanır.",
    "Detector & Threshold rule & F1, in & F1, cross & F1, D & Recall, D & F1, E & F1, ext. & FPR, ext.":
        "Dedektör & Eşik kuralı & F1, aynı & F1, çapraz & F1, D & Duyarlılık, D & F1, E & F1, dış & FPR, dış",
    "Fixed 0.5": "Sabit 0.5",
    "Validation F1": "Doğrulama F1",
    "Pseudo-unseen F1": "Sözde görülmemiş F1",
    "Human 5\\% FPR": "İnsan \\%5 FPR",
    "Human 1\\% FPR": "İnsan \\%1 FPR",
    # coordinator.tex
    "F1 of RoBERTa, the Coordinator and its variants at threshold 0.5 in all scenarios (mean over runs; $\\pm$ std "
    "over seeds for the first two). Gap is Coordinator minus RoBERTa, with a paired source-level bootstrap interval; "
    "* marks Holm-adjusted p $<$ 0.05. ``Unseen stack'' fits the Coordinator on validation texts of generators absent "
    "from training. ``Multi-agent, same sources'' is the reference for the single-agent row.":
        "Bütün senaryolarda eşik 0.5'te RoBERTa, Koordinatör ve türevlerinin F1 değeri (çalıştırmalar üzerinden "
        "ortalama; ilk ikisinde seed'ler üzerinden $\\pm$ std). Fark, Koordinatör eksi RoBERTa; aralık kaynak düzeyinde "
        "eşleştirilmiş bootstrap ile; * Holm düzeltmeli p $<$ 0.05. ``Görülmemiş yığın'' Koordinatör'ü eğitimde olmayan "
        "üreticilerin doğrulama metinleriyle kurar. ``Çok ajan, aynı kaynaklar'' tek ajan satırının referansıdır.",
    "Scenario & RoBERTa & Coord. & Gap & 95\\% CI & Best on val. & Mean logit & Unseen stack & + zero-shot":
        "Senaryo & RoBERTa & Koord. & Fark & \\%95 GA & Doğr. en iyi & Ort. logit & Görülm. yığın & + sıfır örn.",
    "Cross, local": "Çapraz, yerel",
    "Leave-one-out": "Birini dışarıda bırak",
    "D, one generator": "D, tek üretici",
    "E, one generator": "E, tek üretici",
    "Multi-agent, same sources, ABC": "Çok ajan, aynı kaynaklar, ABC",
    "Single-agent, ABC": "Tek ajan, ABC",
    "Writer only, ABC": "Yalnız Yazar, ABC",
    "Paraphrase, other model, ABC": "Yeniden ifade, başka model, ABC",
    "Paraphrase twice, ABC": "İki kez yeniden ifade, ABC",
    "Paraphrase, ABC": "Yeniden ifade, ABC",
    "Back-translation, ABC": "Geri çeviri, ABC",
    "Humanize, ABC": "``İnsanlaştır'' promptu, ABC",
    "Advisory-style prompt, ABC": "Bildiri üslubu promptu, ABC",
    "Few-shot prompt, ABC": "Az örnekli prompt, ABC",
    "25\\% LLM, ABC": "\\%25 LLM, ABC",
    "50\\% LLM, ABC": "\\%50 LLM, ABC",
    "75\\% LLM, ABC": "\\%75 LLM, ABC",
    # robustness.tex and augmented.tex
    "Recall on LLM texts at threshold 0.5 under rewriting, prompt strategy and partial generation, trained on "
    "CISA + ABC (mean $\\pm$ std over three seeds). * marks a recall change from the same excerpts before the change "
    "with Holm-adjusted p $<$ 0.05 (per detector).":
        "Yeniden yazım, prompt stratejisi ve kısmi üretim altında LLM metinlerinde eşik 0.5'te duyarlılık, CISA + ABC "
        "ile eğitim (üç seed üzerinden ortalama $\\pm$ std). *, aynı parçaların değişiklikten önceki hâline göre "
        "duyarlılık değişiminin Holm düzeltmeli p $<$ 0.05 olduğunu gösterir (dedektör başına).",
    "Writer only, no Editor": "Yalnız Yazar, Editör yok",
    "Paraphrase, same model": "Yeniden ifade, aynı model",
    "Paraphrase, other model": "Yeniden ifade, başka model",
    "Paraphrase twice": "İki kez yeniden ifade",
    "Back-translation": "Geri çeviri",
    "``Humanize'' prompt": "``İnsanlaştır'' promptu",
    "Advisory-style prompt": "Bildiri üslubu promptu",
    "Few-shot human examples": "Az örnekli prompt (insan parçaları)",
    "Single agent reading the source": "Kaynağı okuyan tek ajan",
    "Single agent": "Tek ajan",
    "25\\% LLM sentences": "\\%25 LLM cümlesi",
    "50\\% LLM sentences": "\\%50 LLM cümlesi",
    "75\\% LLM sentences": "\\%75 LLM cümlesi",
    "Recall at threshold 0.5 (mean $\\pm$ std over three seeds), ABC training (base) against ABC plus single-agent "
    "and paraphrased texts of training sources (aug.).":
        "Eşik 0.5'te duyarlılık (üç seed üzerinden ortalama $\\pm$ std): ABC ile eğitim (temel) ve ABC'ye eğitim "
        "kaynaklarının tek ajanlı ve yeniden ifade edilmiş metinleri eklenerek eğitim (art.).",
    " & base & aug. & base & aug. & base & aug.": " & temel & art. & temel & art. & temel & art.",
    # cost.tex
    "Cost and performance, one training generator. Inference on one RTX 4060 laptop GPU, three repetitions over a "
    "fixed sample. The Coordinator cost is the sum of its members run one after another; ``Fit'' excludes the LLM "
    "judge, which is not trained.":
        "Maliyet ve başarım, tek eğitim üreticisi. Çıkarım tek bir RTX 4060 dizüstü GPU'da, sabit bir örnek üzerinde "
        "üç tekrar. Koordinatör'ün maliyeti üyelerinin art arda çalışma süresinin toplamıdır; ``Eğitim'' sütunu "
        "eğitilmeyen LLM Yargıç'ı içermez.",
    "Detector & ms/text & Rel. & GPU MiB & Fit s & F1, in & F1, D & AUC, D & F1, E":
        "Dedektör & ms/metin & Oran & GPU MiB & Eğitim s & F1, aynı & F1, D & AUC, D & F1, E",
    # profile.tex
    "Mean text properties by source (multi-agent texts for A--E). First person and agency: share of texts with at "
    "least one first-person pronoun or agency name. Cues: occurrences per 100 words of the word n-grams with the "
    "largest Lexical-LR weights on each side (model trained on A).":
        "Kaynağa göre ortalama metin özellikleri (A--E için çok ajanlı metinler). Birinci şahıs ve kurum adı: en az bir "
        "birinci şahıs zamiri veya kurum adı içeren metinlerin payı. İpuçları: Lexical-LR'nin iki yönde en büyük ağırlığı "
        "verdiği kelime n-gram'larının 100 kelimedeki geçiş sayısı (A ile eğitilen model).",
    "Source & Words & MTLD & Proper nouns & First person & Agency named & LLM-side cues & Human-side cues":
        "Kaynak & Kelime & MTLD & Özel ad & Birinci şahıs & Kurum adı & LLM yönlü ipucu & İnsan yönlü ipucu",
    "Human (Microsoft)": "İnsan (Microsoft)",
    "Human (Mandiant)": "İnsan (Mandiant)",
    # errors.tex
    "Error categories over all seeds: count and share of the errors of each detector. Categories are mutually "
    "exclusive and assigned by the first rule that applies (Section~\\ref{sec:errors}); a text counts once per run in "
    "which it is misclassified. ``All texts'' is the share of the category among all texts of the group, errors or "
    "not, so a category is over-represented among the errors only when its share exceeds this. FN and CISA rows: "
    "models trained on one generator; vendor rows: models trained on ABC.":
        "Bütün seed'ler üzerinden hata kategorileri: her dedektörün hatalarının sayısı ve payı. Kategoriler birbirini "
        "dışlar ve uyan ilk kurala göre atanır (Bölüm~\\ref{sec:errors}); bir metin yanlış sınıflandırıldığı her "
        "çalıştırmada bir kez sayılır. ``Bütün metinler'', kategorinin gruptaki hatalı ya da hatasız bütün metinler "
        "içindeki payıdır; bir kategori ancak hatalar içindeki payı bunu aştığında hatalarda fazla temsil edilir. FN ve "
        "CISA satırları: tek üreticiyle eğitilen modeller; firma satırları: ABC ile eğitilen modeller.",
    "Error & Category & All texts & Lexical-LR & RoBERTa & Coordinator":
        "Hata & Kategori & Bütün metinler & Lexical-LR & RoBERTa & Koordinatör",
    "FN, other generators": "FN, diğer üreticiler",
    "FP, vendor blogs": "FP, firma blogları",
    "first person, still missed": "birinci şahıs, yine de kaçırıldı",
    "cites an agency": "kurum adı geçiyor",
    "unsupported technical detail": "olgularda olmayan teknik ayrıntı",
    "impersonal, human length": "kişisiz, insan uzunluğunda",
    "impersonal, short": "kişisiz, kısa",
    "advice-heavy": "tavsiye ağırlıklı",
    "few proper nouns": "az özel ad",
    "product vulnerability notice": "ürün açığı bildirimi",
    "first person": "birinci şahıs",
    " & other & ": " & diğer & ",
    # contrast.tex
    "Properties of misclassified against correctly classified texts: means of the two groups and Cliff's $\\delta$ "
    "(positive when the misclassified texts have more). A text belongs to the first group when at least half of the "
    "runs misclassify it. * marks a Mann-Whitney test with Benjamini-Hochberg adjusted p $<$ 0.05. D: models trained "
    "on one generator; vendor excerpts: models trained on ABC.":
        "Yanlış ve doğru sınıflandırılan metinlerin özellikleri: iki grubun ortalamaları ve Cliff $\\delta$ değeri "
        "(yanlış sınıflandırılanlarda daha yüksekse pozitif). Bir metin, çalıştırmaların en az yarısında yanlış "
        "sınıflandırıldıysa ilk gruba girer. *, Benjamini-Hochberg düzeltmeli p $<$ 0.05 olan Mann-Whitney testini "
        "gösterir. D: tek üreticiyle eğitilen modeller; firma parçaları: ABC ile eğitilen modeller.",
    "Group & Property & ": "Grup & Özellik & ",
    " & & means & $\\delta$ & means & $\\delta$ & means & $\\delta$":
        " & & ortalamalar & $\\delta$ & ortalamalar & $\\delta$ & ortalamalar & $\\delta$",
    "D texts, missed / caught": "D, kaçan / yakalanan",
    "Vendor excerpts, flagged / passed": "Firma, işaretlenen / geçen",
    " & Texts & ": " & Metin sayısı & ",
    " & Words & ": " & Kelime & ",
    "Type-token ratio": "Tür-örnek oranı (TTR)",
    "Proper-noun share": "Özel ad oranı",
    "Pronoun share": "Zamir oranı",
    "Uses first person (share of texts)": "Birinci şahıslı metin payı",
    "First person per 100 words": "100 kelimede birinci şahıs",
    "LLM-side cues per 100 words": "100 kelimede LLM yönlü ipucu",
    "Human-side cues per 100 words": "100 kelimede insan yönlü ipucu",
    "Identifiers not in the facts": "Olgularda olmayan tanımlayıcı",
    # examples.tex
    "Examples of confident errors (CISA excerpts and LLM output), with the generator, the length and the mean "
    "score over the runs that misclassify the text.":
        "Yüksek güvenle yapılan hatalara örnekler (CISA parçaları ve LLM çıktısı); üretici, uzunluk ve metni yanlış "
        "sınıflandıran çalıştırmalardaki ortalama skorla birlikte.",
    "Error & Opening of the text": "Hata & Metnin başı",
    " words, mean P(LLM) ": " kelime, ortalama P(LLM) ",
    ". Generator ": ". Üretici ",
    # leakage.tex
    "Main results on all test excerpts and without those whose human or LLM text has a near-duplicate in the "
    "training or validation split of the same seed (containment of word 5-grams":
        "Ana sonuçlar, bütün test parçaları üzerinde ve insan ya da LLM metninin aynı seed'in eğitim veya doğrulama "
        "bölmesinde yakın kopyası bulunan parçalar çıkarılarak (kelime 5-gramlarının kapsanma oranı",
    "; strict: ": "; sıkı: ",
    "). Mean test-set size: ": "). Ortalama test kümesi büyüklüğü: ",
    " texts. F1 is mean $\\pm$ std over three seeds.": " metin. F1, üç seed üzerinden ortalama $\\pm$ std.",
    "Setting & Detector & \\multicolumn{3}{c}{F1}": "Senaryo & Dedektör & \\multicolumn{3}{c}{F1}",
    " & & All & No dup. & Strict & All & No dup. & Strict": " & & Tümü & Kopyasız & Sıkı & Tümü & Kopyasız & Sıkı",
    # controls.tex
    "Control trainings, F1 at threshold 0.5 (mean over three seeds and the training generators A, B, C). Under the "
    "two alternative splits the test sets differ from the main ones, and D and E cover fewer of their excerpts.":
        "Kontrol eğitimleri, eşik 0.5'te F1 (üç seed ve A, B, C eğitim üreticileri üzerinden ortalama). İki alternatif "
        "bölmede test kümeleri ana deneydekinden farklıdır ve D ile E bu kümelerin daha az parçasını kapsar.",
    "Split & Detector & In & Cross & D & E": "Bölme & Dedektör & Aynı & Çapraz & D & E",
    "By report (used)": "Rapor bazlı (kullanılan)",
    "Near-duplicate reports joined": "Yakın kopya raporlar birleşik",
    "By excerpt (leaky)": "Parça bazlı (sızdıran)",
    "By report, labels shuffled": "Rapor bazlı, etiketler karışık",
    # newgen.tex
    "Errors, ranking and probability quality on the unseen generators D and E, next to the in-generator reference. "
    "FPR is measured on the CISA excerpts of each test set and FNR on the LLM texts. ECE and Brier use the "
    "uncalibrated scores. The last two columns use the threshold at the 95th percentile of validation human scores. "
    "Mean $\\pm$ std over three seeds.":
        "Görülmemiş üreticiler D ve E'de hata oranları, sıralama ve olasılık kalitesi; karşılaştırma için aynı üretici "
        "sonuçları. FPR her test kümesinin CISA parçalarında, FNR LLM metinlerinde ölçülür. ECE ve Brier kalibre "
        "edilmemiş skorlarla hesaplanır. Son iki sütunda eşik, doğrulama insan skorlarının 95. yüzdeliğidir. Üç seed "
        "üzerinden ortalama $\\pm$ std.",
    " & & \\multicolumn{6}{c}{Threshold 0.5, raw scores} & \\multicolumn{2}{c}{Human 5\\% rule}":
        " & & \\multicolumn{6}{c}{Eşik 0.5, ham skorlar} & \\multicolumn{2}{c}{İnsan \\%5 kuralı}",
    "Test & Detector & F1 & FPR & FNR & ROC-AUC & ECE & Brier & FPR & FNR":
        "Test & Dedektör & F1 & FPR & FNR & ROC-AUC & ECE & Brier & FPR & FNR",
}
