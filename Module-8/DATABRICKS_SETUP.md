# Running Module 8 & 9 on Databricks Free Edition

Native PySpark on Windows hits an unresolved local-mode bug (the Python
worker process gets silently killed the instant a real task runs — see git
history for the details we ran into). Rather than fight that, both modules
now target **Databricks Free Edition** (the current name for what used to be
called Databricks Community Edition): a hosted Spark environment with no
local Java/PySpark install required.

Both notebooks already load their data via `seaborn`
(`sns.load_dataset("titanic")`) or from in-notebook Python collections —
there's no CSV to upload and no DBFS path wrangling needed to get started.

---

## 1. Create a Databricks Free Edition account

Go to [databricks.com](https://www.databricks.com) and sign up for the
**Free Edition**. It's a separate signup path from a paid trial — look for
"Free Edition" specifically, not "Try Databricks" (which starts a 14-day
trial of the paid product). Free Edition accounts don't expire and don't
require a credit card.

## 2. Compute

Free Edition provisions serverless compute automatically — there's normally
no cluster to size or configure manually. If your workspace still shows a
cluster picker, create the smallest available all-purpose cluster with a
current runtime (Spark 3.5+ / Databricks Runtime 14+ LTS or newer).

One thing to know either way: compute **auto-terminates after a period of
idle time**. If a notebook errors with something like "cluster not
attached" after a break, just re-run the first cell — it'll spin back up.

## 3. Import the notebook

In the Databricks workspace sidebar: **Workspace → your folder → Import**,
then upload the `.ipynb` file directly (Databricks imports Jupyter notebooks
natively — no conversion needed). Do this for both
`Module8_SparkTitanic.ipynb` and `Module_9_Spark_RDD.ipynb`.

## 4. Libraries

`pandas` and common ML libraries are preinstalled on Databricks Runtime.
`seaborn` may not be — if `import seaborn` fails, add a cell at the top:
```python
%pip install seaborn
```
(Restart the Python kernel with `dbutils.library.restartPython()` after, if
prompted.)

## 5. `SparkSession`

Databricks notebooks already have a `spark` variable in scope. The
notebooks' existing `SparkSession.builder.appName(...).getOrCreate()` calls
still work fine on Databricks — they just return the existing session
rather than creating a new one — so no code changes are needed there.

## 6. Module 9's RDD save/load

Module 9 writes an RDD to disk and reads it back. It now targets an
explicit `dbfs:/tmp/...` path with `dbutils.fs.rm` for cleanup, instead of
a local relative path with `shutil` — that avoids relying on ambiguous
relative-path resolution and works on both classic and serverless compute.

---

## Running it

Once imported and attached to compute, run each notebook top to bottom the
same way you would locally. No JDK, WSL2, `winutils`, or `JAVA_HOME` setup
is needed — that's the whole point of moving here.
