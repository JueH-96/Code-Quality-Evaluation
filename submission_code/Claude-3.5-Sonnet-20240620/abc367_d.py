# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i + 1] = prefix_sum[i] + A[i]

total_sum = prefix_sum[N]

count = 0
for s in range(N):
    for t in range(s + 1, N + s):
        t %= N
        if s == t:
            continue
        if t > s:
            steps = prefix_sum[t] - prefix_sum[s]
        else:
            steps = total_sum - (prefix_sum[s] - prefix_sum[t])
        if steps % M == 0:
            count += 1

print(count)