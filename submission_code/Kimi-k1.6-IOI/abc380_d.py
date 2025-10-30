S = input().strip()
Q = int(input())
K_list = list(map(int, input().split()))
L = len(S)
ans = []

for K in K_list:
    m = 0
    current_length = L
    while current_length < K:
        current_length *= 2
        m += 1
    pos = K
    flip = 0
    cl = current_length  # equals L * 2^m
    for _ in range(m):
        half = cl // 2
        if pos > half:
            pos -= half
            flip += 1
        cl = half
    c = S[pos-1]
    if flip % 2 == 1:
        c = c.swapcase()
    ans.append(c)

print(' '.join(ans))