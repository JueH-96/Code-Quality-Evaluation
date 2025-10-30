N, M = map(int, input().split())
B = list(map(int, input().split()))
W = list(map(int, input().split()))

# Sort in descending order
B.sort(reverse=True)
W.sort(reverse=True)

# Compute prefix sums
B_prefix = [0] * (N + 1)
for i in range(N):
    B_prefix[i + 1] = B_prefix[i] + B[i]

W_prefix = [0] * (M + 1)
for i in range(M):
    W_prefix[i + 1] = W_prefix[i] + W[i]

# Compute max_B_suffix[w] = max(B_prefix[b] for b in range(w, N+1))
max_B_suffix = [0] * (N + 1)
for w in range(N, -1, -1):
    if w == N:
        max_B_suffix[w] = B_prefix[N]
    else:
        max_B_suffix[w] = max(B_prefix[w], max_B_suffix[w + 1])

# Find the answer
answer = 0
for w in range(M + 1):
    if w <= N:  # We can choose at least w black balls
        answer = max(answer, W_prefix[w] + max_B_suffix[w])

print(answer)