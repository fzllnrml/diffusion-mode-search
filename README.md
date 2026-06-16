# Поиск мод распределений с помощью диффузионных score-моделей

Поиск мод распределений, заданных обученными диффузионными score-моделями.

Методы проверялись на смесях гауссовских распределений размерности 1, 2, 10, 30 и 50. Отдельный эксперимент проведён на наборе масс-цитометрии Levine 13D.

## Структура проекта

- `src/` — модели, алгоритмы поиска мод, метрики и вспомогательные функции
- `configs/` — конфигурации экспериментов
- `scripts/` — запуск экспериментов на синтетических данных
- `checkpoints/` — сохранённые модели для синтетических экспериментов
- `results_1d/`, `results_2d/`, `results_10d/`, `results_30d/`, `results_50d/` — результаты экспериментов
- `results_pareto_validation_all/` — дополнительные проверки качества score-модели и найденных мод
- `levine/` — отдельный проект для эксперимента Levine 13D
- `results_levine13/` — результаты эксперимента Levine 13D.

## Установка

Код запускался на Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Выбор устройства значения: `auto`, `cpu`, `mps` и `cuda`. 
На Mac с Apple Silicon  `mps`.

## Запуск одного эксперимента

Команды нужно выполнять из корня репозитория.

Пример запуска метода `v3f2` для 10-мерной смеси:

```bash
python3 scripts/run_experiment.py \
  --config configs/presets/dim10.yaml \
  --method v3f2
```

Основные методы:

- `v1`;
- `v2`;
- `v3f1`;
- `v3f2`;
- `v3f3`.

Пути к checkpoint и результатам можно изменить с помощью `--checkpoint-dir` и `--output-dir`.

## Серии экспериментов

Стандартные серии для 10D при `eps_hit = 2`:

```bash
python3 scripts/run_all_experiments.py \
  --config configs/presets/dim10_eps2.yaml \
  --experiment all \
  --method v3all
```

Вместо `all` можно указать отдельную серию:

- `sweep_k` — изменение числа компонент смеси;
- `sweep_starts` — изменение числа стартовых точек;
- `sweep_nfe` — изменение вычислительного бюджета.

## Зависимость от размера популяции

```bash
python3 scripts/run_population_experiment.py \
  --config configs/presets/dim10_eps2.yaml \
  --method v3 \
  --K 4 \
  --n-particles 10,20,30,50,75,100
```

## Чувствительность к порогу ошибки

```bash
python3 scripts/run_eps_sensitivity_all.py \
  --config configs/presets/dim10.yaml \
  --methods v2,v3f1,v3f2,v3f3,b0,b10 \
  --k-values 2,3,4,5,6,7 \
  --eps 1.0,1.5,2.0,2.5,3.0 \
  --checkpoint-dir checkpoints \
  --output-dir results_10d/eps_sensitivity_all
```

Для 30D и 50D используются конфигурации:

```text
configs/presets/dim30.yaml
configs/presets/dim50.yaml
```

## Проверочные эксперименты

Проверка checkpoint:

```bash
python3 scripts/run_pareto_validation_experiments.py \
  --profile full \
  --checkpoint-dir checkpoints \
  --check-only
```


Полный запуск:

```bash
python3 scripts/run_pareto_validation_experiments.py \
  --profile full \
  --checkpoint-dir checkpoints \
  --output-dir results_pareto_validation_all
```

## Эксперимент Levine 13D

Этот эксперимент запускался отдельно.

```bash
cd levine
```

Подготовка данных, обучение моделей и основные команды описаны в файле levine/README.md.

В репозитории есть 2 чекпоинта:

- `levine/checkpoints/levine13_all_100k.pt` — модель, обученная на всех клетках
- `levine/checkpoints/levine13_labeled_100k.pt` — модель, обученная только на размеченных клетках.

## Результаты

Результаты уже сохранены в репозитории. Повторно обучать модели для их просмотра не требуется.

Основные директории:

- `results_10d/`, `results_30d/`, `results_50d/` — эксперименты на синтетических смесях;
- `results_pareto_validation_all/` — дополнительные эксперименты
- `results_levine13/full_sweep/summary.csv` — сводная таблица методов на Levine 13D
- `results_levine13/baseline_sample_sweep/summary.csv` — результаты базового метода
- `results_levine13/result_main_clean/` — подробные результаты основного эксперимента
- `results_levine13/additional_complete_fixed_checkpoint/` — дополнительные проверки Levine 13D.

ля анализа с результатов удобнее смотреть итоговые CSV-файлы.

## Данные Levine 13D

Использована подготовленная версия набора Levine 13D из репозитория:

https://github.com/lmweber/benchmark-data-Levine-13-dim

Набор содержит 167 044 клетки и 13 маркеров. Для 81 747 клеток известны экспертные метки 24 популяций, остальные клетки не размечены.

Использованная работа:

> Levine J. H. et al. Data-Driven Phenotypic Dissection of AML Reveals Progenitor-like Cells that Correlate with Prognosis. Cell. 2015. Vol. 162. P. 184–197. https://doi.org/10.1016/j.cell.2015.05.047



## Доп Замечания

- параметры экспериментов находятся в YAML-конфигурациях и скриптах;
- обученные модели для синтетических данных находятся в `checkpoints/`;
- seed поиска сохраняется в названиях файлов и итоговых таблицах;
- для старых checkpoint Levine seed обучения не сохранился;
- время работы зависит от устройства и версии библиотек.
