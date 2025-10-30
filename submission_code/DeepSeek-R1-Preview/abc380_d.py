S = input().strip()
Q = int(input())
K_list = list(map(int, input().split()))

len_S = len(S)
result = []

for K in K_list:
    pos = K - 1
    quotient = pos // len_S
    count = bin(quotient).count('1')
    c = S[pos % len_S]
    if count % 2 == 1:
        c = c.swapcase()
    result.append(c)

print(' '.join(result))