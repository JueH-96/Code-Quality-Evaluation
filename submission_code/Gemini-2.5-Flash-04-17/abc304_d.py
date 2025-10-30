import sys
from bisect import bisect_right
from collections import Counter

# Read input
# W and H define the cake dimensions, but their large values (up to 10^9)
# mean we don't use them directly for array indexing or iteration.
# Their relevance is in defining the range for p_i, q_i, a_i, b_i.
# We only need the relative positions of strawberries and cut lines.
# So, we can read and discard W and H.
# Using input() might be slightly slower than sys.stdin.readline for very large inputs,
# but for N, A, B up to 2*10^5, standard input is usually sufficient.
# input = sys.stdin.readline # Uncomment for potentially faster input reading

input() # Read W and H and discard

N = int(input())

# Read N strawberry coordinates (p_i, q_i)
# Store them in a list of tuples.
strawberries = []
for _ in range(N):
    p, q = map(int, input().split())
    strawberries.append((p, q))

A = int(input())
# Read A vertical cut lines a_1, ..., a_A. Guaranteed to be sorted: 0 < a_1 < ... < a_A < W.
a = list(map(int, input().split()))

B = int(input())
# Read B horizontal cut lines b_1, ..., b_B. Guaranteed to be sorted: 0 < b_1 < ... < b_B < H.
b = list(map(int, input().split()))

# We want to count how many strawberries fall into each rectangular piece.
# The cuts divide the cake into (A+1) * (B+1) pieces.
# A vertical cuts at a_1, ..., a_A divide the x-range [0, W] into A+1 intervals:
# [0, a_1], [a_1, a_2], ..., [a_A, W]. Let's index these intervals 0 to A.
# The interval [a_i, a_{i+1}] (with a_0=0, a_{A+1}=W) corresponds to vertical strip i.
# Similarly, B horizontal cuts at b_1, ..., b_B divide the y-range [0, H] into B+1 intervals:
# [0, b_1], [b_1, b_2], ..., [b_B, H]. Let's index these intervals 0 to B.
# The interval [b_j, b_{j+1}] (with b_0=0, b_{B+1}=H) corresponds to horizontal strip j.
# A piece is the intersection of a vertical strip i and a horizontal strip j, forming a rectangle.
# This piece can be identified by the pair of indices (i, j).
# Since strawberries are not on the boundaries (0, W, 0, H) or the cut lines (a_i, b_j),
# a strawberry (p, q) is strictly inside the piece (i, j) if a_i < p < a_{i+1} and b_j < q < b_{j+1}.

# We can use bisect_right to find the index of the interval a coordinate falls into.
# For a value p and a sorted list of cuts `a`, bisect_right(a, p) returns the index `k`
# such that all elements `a[:k]` are <= p and all elements `a[k:]` are > p.
# Since p is not in `a`, this is the count of elements in `a` less than p.
# If p < a_1, bisect_right(a, p) = 0. This is vertical strip 0.
# If a_i < p < a_{i+1}, bisect_right(a, p) = i+1. This is vertical strip i+1? No, wait.
# Let's redefine vertical strip index i:
# i=0: (0, a_1)
# i=1: (a_1, a_2)
# ...
# i=k: (a_k, a_{k+1})
# ...
# i=A: (a_A, W)
# bisect_right(a, p):
# If p < a_1, bisect_right returns 0. This should be strip index 0. Correct.
# If a_1 < p < a_2, bisect_right returns 1. This should be strip index 1. Correct.
# If a_i < p < a_{i+1}, bisect_right returns i+1. This should be strip index i+1. Correct.
# If a_A < p < W, bisect_right returns A. This should be strip index A. Correct.
# Yes, bisect_right(a, p) gives the correct 0-based index of the vertical strip.
# Similarly, bisect_right(b, q) gives the correct 0-based index of the horizontal strip.

# Use a Counter to count strawberries in each piece (identified by (vertical_index, horizontal_index)).
counts = Counter()

# Process each strawberry
for p, q in strawberries:
    # Find the vertical strip index
    vertical_index = bisect_right(a, p)
    # Find the horizontal strip index
    horizontal_index = bisect_right(b, q)
    # Increment the count for this piece
    counts[(vertical_index, horizontal_index)] += 1

# Total number of pieces created by the cuts
total_pieces = (A + 1) * (B + 1)

# Determine the minimum and maximum number of strawberries in any piece.

# Minimum:
# If there is at least one piece with zero strawberries, the minimum is 0.
# A piece has zero strawberries if its (vertical_index, horizontal_index) pair is not
# present as a key in the `counts` dictionary.
# The total number of distinct piece indices is `total_pieces`.
# The number of pieces with at least one strawberry is `len(counts)`.
# If `len(counts) < total_pieces`, there are some pieces missing from `counts`, which means they are empty.
if len(counts) < total_pieces:
    min_strawberries = 0
else:
    # If `len(counts) == total_pieces`, it means every piece (identified by (i, j) from 0..A, 0..B)
    # has at least one strawberry (its key is in `counts`).
    # The minimum count is the smallest value among all counts in the dictionary.
    # Since N >= 1, `counts` is not empty, so min(counts.values()) is safe.
    min_strawberries = min(counts.values())

# Maximum:
# The maximum number of strawberries in a piece is simply the highest count found
# among all the pieces that contain at least one strawberry.
# Since N >= 1, there is at least one strawberry, so `counts` is not empty.
max_strawberries = max(counts.values())

# Print the result
print(min_strawberries, max_strawberries)