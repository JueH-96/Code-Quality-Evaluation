import sys
sys.setrecursionlimit(1000100)

data = sys.stdin.read().split()
N = int(data[0])
K = int(data[1])
X = int(data[2])
T = list(map(int, data[3:3+N]))

prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i + 1] = prefix_sum[i] + T[i]

DUMMY_S = -1000000000000  # -10**12

memo = {}

def min_diss(pos, last_S):
    if (pos, last_S) in memo:
        return memo[(pos, last_S)]
    if pos == N:
        cost = 0
    else:
        cost = 2 * 10**14  # large number
        end_max = min(pos + K - 1, N - 1)
        for m in range(pos, end_max + 1):
            if pos == 0:
                S = T[m]
            else:
                S = max(T[m], last_S + X)
            size_batch = m - pos + 1
            sum_T_batch = prefix_sum[m + 1] - prefix_sum[pos]
            diss_added = size_batch * S - sum_T_batch
            cost_m = diss_added + min_diss(m + 1, S)
            if cost_m < cost:
                cost = cost_m
    memo[(pos, last_S)] = cost
    return cost

result = min_diss(0, DUMMY_S)
print(result)