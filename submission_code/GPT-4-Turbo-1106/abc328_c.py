# Read the input
N, Q = map(int, input().split())
S = input().strip()
queries = [tuple(map(int, input().split())) for _ in range(Q)]

# Precompute the number of consecutive equal characters up to each position
consecutive_counts = [0] * (N + 1)
for i in range(1, N):
    consecutive_counts[i + 1] = consecutive_counts[i] + (S[i] == S[i - 1])

# Answer the queries
for l, r in queries:
    # Subtract the precomputed value at the start of the range from the value at the end
    # to get the number of consecutive equal characters in the range
    print(consecutive_counts[r] - consecutive_counts[l])