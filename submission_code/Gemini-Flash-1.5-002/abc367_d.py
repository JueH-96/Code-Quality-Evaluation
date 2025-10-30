# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))

total_steps = sum(a)
prefix_sums = [0] * (n + 1)
for i in range(n):
    prefix_sums[i + 1] = prefix_sums[i] + a[i]

count = 0
for s in range(1, n + 1):
    for t in range(1, n + 1):
        if s == t:
            continue
        steps = 0
        if t > s:
            steps = prefix_sums[t] - prefix_sums[s]
        else:
            steps = total_steps - (prefix_sums[s] - prefix_sums[t])
        if steps % m == 0:
            count += 1

print(count)