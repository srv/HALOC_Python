# HALOC-implemented-in-Python
HALOC algorithm for image hashing and comparison. See https://github.com/srv/libhaloc for the original HALOC.
It creates the 3 orthogonal vectors, it hashes one query image using HALOC (see https://github.com/srv/libhaloc) and all images of a database, and outputs the distance in terms of L1-norm between the query hash and the hash of each image of the database.
