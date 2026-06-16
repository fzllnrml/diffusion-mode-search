# Эксперимент Levine 13D

Отдельный эксперимент на данных масс-цитометрии Levine 13D. Он запускался независимо от экспериментов на синтетических смесях, поэтому скрипты, модели и копия исходного кода отдельно сохранены.

Команды ниже нужно выполнять из директории `levine`:


## Содержимое директории

- `data/` — исходные и подготовленные данные
- `checkpoints/` — обученные диффузионные модели
- `scripts/` — подготовка данных, обучение и основные эксперименты
- `scripts_additional/` — дополнительные проверки
- `src/` — версия алгоритмов, использованная в этих запусках.

Результаты в `../results_levine13/`.

## Подготовка данных

В директории `data/` должны находиться файлы с данными:

```text
data/Levine_13dim.txt
data/population_names_Levine_13dim.txt
```

Для подготовки данных:

```bash
PYTHONPATH=. python3 scripts/prepare_levine13.py
```

Будет создан файл:

```text
data/levine13/levine13_processed.npz
```

## Обучение моделей

Обучать модели повторно не нужно. checkpoint уже в `checkpoints/`.

Обучение на всех клетках:

```bash
PYTHONPATH=. python3 scripts/train_levine13_diffusion.py \
  --mode all \
  --steps 100000 \
  --batch-size 512 \
  --device auto \
  --out checkpoints/levine13_all_100k.pt
```

Обучение только на размеченных клетках:

```bash
PYTHONPATH=. python3 scripts/train_levine13_diffusion.py \
  --mode labeled \
  --steps 100000 \
  --batch-size 512 \
  --device auto \
  --out checkpoints/levine13_labeled_100k.pt
```

## Основной запуск методов

```bash
PYTHONPATH=. python3 scripts/run_levine13_full_sweep.py \
  --device auto \
  --checkpoints all=checkpoints/levine13_all_100k.pt,labeled=checkpoints/levine13_labeled_100k.pt \
  --methods v2,v3f1,v3f2,v3f3 \
  --r-values 0.5,0.8,1.0,1.2,1.5,2.0 \
  --seeds 0 \
  --data-path data/levine13/levine13_processed.npz \
  --pop-path data/population_names_Levine_13dim.txt \
  --out-dir ../results_levine13/full_sweep
```

В основном эксперименте использовались:

- 300 частиц
- 30 стартовых точек
- 20 шагов подъёма
- 100 шагов уточнения
- 50 DDIM-шагов
- 100 ближайших соседей
- порог чистоты 0.90
- допустимая доля неразмеченных соседей 0.20.

## Базовый метод

Проверялись два варианта:

- `b0` — генерация точек обратным диффузионным процессом и последующая кластеризация
- `b10` — тот же метод с десятью дополнительными шагами уточнения.

Команда запуска:

```bash
PYTHONPATH=. python3 scripts/run_levine13_baseline_sample_sweep.py \
  --device auto \
  --checkpoints all=checkpoints/levine13_all_100k.pt,labeled=checkpoints/levine13_labeled_100k.pt \
  --n-samples-values 500,1000,2000,5000,10000 \
  --r-values 0.5,0.8,1.0,1.2,1.5,2.0,2.5,3.0,4.0 \
  --refine-steps 10 \
  --refine-alpha 0.01 \
  --seeds 0 \
  --out-dir ../results_levine13/baseline_sample_sweep
```

## Диагностика

Проверка размеченных популяций и чистоты ближайших соседей:

```bash
PYTHONPATH=. python3 scripts/diagnose_levine13.py
```

Проверка выборки из моделью:

```bash
PYTHONPATH=. python3 scripts/eval_levine13_generation.py \
  --ckpt checkpoints/levine13_labeled_100k.pt \
  --n 5000 \
  --device auto
```


## Дополнительные проверки

В `scripts_additional/` находятся проверки:

- чувствительности результатов к параметрам разметки
- покрытия отдельных клеток
- базового метода mean shift в пространстве данных
- устойчивости найденных кандидатов
- различий между реальными и сгенерированными данными
- целостности сохранённых результатов.

Проверка чувствительности разметки:

```bash
PYTHONPATH=. python3 scripts_additional/run_annotation_sensitivity.py \
  --modes-glob '../results_levine13/result_main_clean/full_sweep/modes_*.npy' \
  --out-dir ../results_levine13/additional_complete_fixed_checkpoint/new_results_reannotation
```

Проверка покрытия клеток:

```bash
PYTHONPATH=. python3 scripts_additional/run_cell_level_coverage.py \
  --modes '../results_levine13/result_main_clean/full_sweep/modes_*.npy' \
  --out-dir ../results_levine13/additional_complete_fixed_checkpoint/cell_level_coverage
```

Проверка устойчивости найденных кандидатов:

```bash
PYTHONPATH=. python3 scripts_additional/run_modality_stability.py \
  --modes '../results_levine13/result_main_clean/full_sweep/modes_*.npy' \
  --checkpoint checkpoints/levine13_labeled_100k.pt \
  --device auto \
  --out-dir ../results_levine13/additional_complete_fixed_checkpoint/modality_stability
```

Проверка структуры итоговых файлов:

```bash
PYTHONPATH=. python3 scripts_additional/verify_levine13_outputs.py \
  --roots ../results_levine13 \
  --report ../results_levine13/additional_complete_fixed_checkpoint/verification_report.txt
```


## Источник данных

Использована версия Levine 13D из репозитория:

https://github.com/lmweber/benchmark-data-Levine-13-dim

Набор содержит 167 044 клетки, 13 маркеров и экспертные метки для 24 клеточных популяций. Размечены 81 747 клеток, остальные 85 297 клеток не имеют экспертной метки.

Работа:

> Levine J. H. et al. Data-Driven Phenotypic Dissection of AML Reveals Progenitor-like Cells that Correlate with Prognosis. Cell. 2015. Vol. 162. P. 184–197. https://doi.org/10.1016/j.cell.2015.05.047

