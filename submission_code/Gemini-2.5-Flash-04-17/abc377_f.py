# Helper function to count squares (i, j) with 1<=i,j<=N and i+j = s
def count_sum(s, N):
    # Pairs (i, j) with i+j=s. i goes from max(1, s-N) to min(N, s-1).
    # Number of integer points in [a, b] is max(0, b - a + 1).
    return max(0, min(N, s - 1) - max(1, s - N) + 1)

# Helper function to count squares (i, j) with 1<=i,j<=N and i-j = d
def count_diff(d, N):
    # Pairs (i, j) with i-j=d. j goes from max(1, 1-d) to min(N, N-d).
    # Number of integer points in [a, b] is max(0, b - a + 1).
    return max(0, min(N, N - d) - max(1, 1 - d) + 1)


# Read input
N, M = map(int, input().split())

R_bad = set()
C_bad = set()
S_bad = set()
D_bad = set()

for _ in range(M):
    r, c = map(int, input().split())
    R_bad.add(r)
    C_bad.add(c)
    S_bad.add(r + c)
    D_bad.add(r - c)

# Inclusion-Exclusion Principle to count invalid squares |A U B U C U D|
# A = {(i, j) | i in R_bad, 1<=j<=N}
# B = {(i, j) | j in C_bad, 1<=i<=N}
# C = {(i, j) | i+j in S_bad, 1<=i,j<=N}
# D = {(i, j) | i-j in D_bad, 1<=i,j<=N}
# |A U B U C U D| = Sum |.| - Sum |.\cap.| + Sum |.\cap.\cap.| - |.\cap.\cap.\cap.|

invalid_count = 0

# Term 1: Sum of sizes of individual sets
# |A|
invalid_count += len(R_bad) * N
# |B|
invalid_count += len(C_bad) * N
# |C|
invalid_count += sum(count_sum(s, N) for s in S_bad)
# |D|
invalid_count += sum(count_diff(d, N) for d in D_bad)

# Term 2: Sum of sizes of pairwise intersections (subtract)
pairwise_intersections = 0

# |A \cap B|
pairwise_intersections += len(R_bad) * len(C_bad)
# |A \cap C|
pairwise_intersections += sum(1 for r in R_bad for s in S_bad if 1 <= s - r <= N)
# |A \cap D|
pairwise_intersections += sum(1 for r in R_bad for d in D_bad if 1 <= r - d <= N)
# |B \cap C|
pairwise_intersections += sum(1 for c in C_bad for s in S_bad if 1 <= s - c <= N)
# |B \cap D|
pairwise_intersections += sum(1 for c in C_bad for d in D_bad if 1 <= c + d <= N)
# |C \cap D|
pairwise_intersections += sum(1 for s in S_bad for d in D_bad if (s % 2 == d % 2) and 1 <= (s + d) // 2 <= N and 1 <= (s - d) // 2 <= N)

invalid_count -= pairwise_intersections

# Term 3: Sum of sizes of triple intersections (add)
triple_intersections = 0

# |A \cap B \cap C|
triple_intersections += sum(1 for r in R_bad for c in C_bad if (r + c) in S_bad)
# |A \cap B \cap D|
triple_intersections += sum(1 for r in R_bad for c in C_bad if (r - c) in D_bad)
# |A \cap C \cap D|
triple_intersections += sum(1 for s in S_bad for d in D_bad if (s % 2 == d % 2) and 1 <= (s + d) // 2 <= N and 1 <= (s - d) // 2 <= N and ((s + d) // 2) in R_bad)
# |B \cap C \cap D|
triple_intersections += sum(1 for s in S_bad for d in D_bad if (s % 2 == d % 2) and 1 <= (s + d) // 2 <= N and 1 <= (s - d) // 2 <= N and ((s - d) // 2) in C_bad)

invalid_count += triple_intersections

# Term 4: Size of quadruple intersection (subtract)
quadruple_intersection = 0

# |A \cap B \cap C \cap D|
quadruple_intersection += sum(1 for r in R_bad for c in C_bad if (r + c) in S_bad and (r - c) in D_bad)

invalid_count -= quadruple_intersection

# Total squares
total_squares = N * N

# Number of safe squares = Total squares - Number of invalid squares
# The invalid_count calculation correctly excludes the M occupied squares from being counted as "safe",
# because any occupied square (a_k, b_k) violates at least one (in fact, all four) safety conditions
# based on properties derived from those same pieces (a_k is in R_bad, etc.) if M > 0.
safe_squares = total_squares - invalid_count

print(safe_squares)