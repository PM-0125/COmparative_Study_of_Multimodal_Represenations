# 📊 DATASET CARDS — Comparative Study of Multimodal Representations

## Amazon Reviews

- **Files:** `train.csv`, `test.csv`
- **Columns:**
    - `full_text`: Concatenated review title and review text (lowercased, cleaned).
    - `label`: Sentiment class (1=Negative, 2=Positive).
- **Notes:** No missing values. ~3.6M train, 400k test.

---

## FashionAI

- **Files:** `train.csv`, `val.csv`
- **Columns:**
    - `image_path`: Absolute path to product image.
    - `description`: Cleaned product description.
    - `label`: Product category (e.g. "Jeans", "Tops", etc.).
- **Notes:** All rows have valid image files and labels.

---

## MovieLens 20M

- **Files:** `train.csv`, `val.csv`, `test.csv`
- **Columns:**
    - `userId`, `movieId`, `rating`, `timestamp`: Original MovieLens fields.
    - `user_idx`, `movie_idx`: Contiguous indices for embedding layers.
    - `title`: Movie title.
    - `genres`: Pipe-separated list of genres.
    - `tag`: Concatenated user tags per movie.
- **Additional files:**
    - `user2idx.csv`, `movie2idx.csv`: Mapping tables.
    - `movie_genome_vectors.csv`: 1129-dim tag embedding per movie.
    - `movie_similarity_edges.csv`: (movie1, movie2, similarity score) for semantic graph.
    - `edge_list.csv`: (user_idx, movie_idx) interactions.
- **Notes:** No missing values. User/movie indices align with embedding layers. Graph features for advanced modeling.

---

## Data Integrity

- **Checked for missing values and label balance.**
- **Label distributions are available in the scripts.**
- **All splits are randomized with fixed seed (42) for reproducibility.**

---

*Last updated: [24.06.2025] by [Pranjul Mishra]*
