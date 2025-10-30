import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1
S = data[index]
index += 1

# Compute prefix sum of equal adjacent pairs
prefix_sum = [0]
for i in range(N - 1):
    val = 1 if S[i] == S[i + 1] else 0
    prefix_sum.append(prefix_sum[-1] + val)

# Now process Q queries
for _ in range(Q):
    l = int(data[index])
    index += 1
    r = int(data[index])
    index += 1
    ans = prefix_sum[r - 1] - prefix_sum[l - 1]
    print(ans)