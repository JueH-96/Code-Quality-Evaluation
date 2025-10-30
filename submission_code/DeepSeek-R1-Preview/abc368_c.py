n, *rest = map(int, open(0).read().split())
h_list = rest[:n]

sum_table = [
    [0, 1, 2],   # m=0
    [0, 1, 4],   # m=1
    [0, 3, 4],   # m=2
]

current_t = 0
for h in h_list:
    m = current_t % 3
    min_k = float('inf')
    for r in range(3):
        s = sum_table[m][r]
        if s >= h:
            k = r
        else:
            required = h - s
            a = (required + 4) // 5
            k = 3 * a + r
        if k < min_k:
            min_k = k
    current_t += min_k

print(current_t)