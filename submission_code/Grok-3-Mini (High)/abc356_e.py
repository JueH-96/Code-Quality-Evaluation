import sys
data = sys.stdin.read().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))
A.sort()

MAX_A = 1000010
ft = [0] * MAX_A

def ft_update(idx, delta):
    while idx < MAX_A:
        ft[idx] += delta
        idx += idx & -idx

def ft_query(idx):
    sum_q = 0
    i = idx
    while i > 0:
        sum_q += ft[i]
        i -= i & -i
    return sum_q

def ft_range_sum(L, R):
    if L > R or L < 1 or R < 1:
        return 0
    return ft_query(R) - ft_query(L - 1)

ans = 0

for j in range(N):
    S = A[j]
    res = 0
    v = 1
    while v <= S:
        k = S // v
        x_max = S // k
        num_elements = ft_range_sum(v, x_max)
        res += k * num_elements
        v = x_max + 1
    ans += res
    ft_update(A[j], 1)

print(ans)