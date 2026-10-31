# Datasets

The datasets aren't stored in this repository because they're large and belong to their publishers. Put these three files in this `data/` folder:

| File | Dataset | Rows | Get it from |
| --- | --- | --- | --- |
| `training.1600000.processed.noemoticon.csv` | Sentiment140 | 1,600,000 tweets | [Kaggle: kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) |
| `spam.csv` | SMS Spam Collection | 5,572 SMS | [Kaggle: uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |
| `enron_spam_data.csv` | Enron-Spam | 33,716 emails | [GitHub: MWiechmann/enron_spam_data](https://github.com/MWiechmann/enron_spam_data) (inside `enron_spam_data.zip`) |

## Option 1: automatic download

```bash
python -m scripts.download_data
```

If Kaggle asks you to sign in, create an API token once:

1. Sign in at kaggle.com, open **Settings**, and under **API** click **Create New Token**. This downloads `kaggle.json`.
2. Move it to `C:\Users\<you>\.kaggle\kaggle.json` on Windows, or `~/.kaggle/kaggle.json` on macOS and Linux.
3. Run the download command again.

## Option 2: download in your browser

Open each link above, click **Download**, unzip, and copy the CSV file here with the exact file name shown in the table.

## Citations

- Go, A., Bhayani, R. & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision.* Stanford University.
- Almeida, T. A., Gómez Hidalgo, J. M. & Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering.* ACM DocEng.
- Metsis, V., Androutsopoulos, I. & Paliouras, G. (2006). *Spam Filtering with Naive Bayes: Which Naive Bayes?* CEAS. Preprocessed CSV by Marcel Wiechmann.
