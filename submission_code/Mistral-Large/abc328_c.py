import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
S = data[2]
queries = []
index = 3

for _ in range(Q):
    l = int(data[index])
    r = int(data[index + 1])
    queries.append((l, r))
    index += 2

# Precompute the number of consecutive pairs for each position
consecutive_pairs = [0] * (N + 1)

for i in range(1, N):
    if S[i - 1] == S[i]:
        consecutive_pairs[i + 1] = consecutive_pairs[i] + 1
    else:
        consecutive_pairs[i + 1] = consecutive_pairs[i]

results = []

for l, r in queries:
    results.append(consecutive_pairs[r] - consecutive_pairs[l])

sys.stdout.write("
".join(map(str, results)) + "
")