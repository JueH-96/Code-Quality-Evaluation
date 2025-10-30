S = input().strip()
Q = int(input())
K_list = list(map(int, input().split()))

original_len = len(S)
result = []

for K in K_list:
    current_k = K - 1
    flips = 0
    L = original_len
    while current_k >= L:
        current_k -= L
        flips += 1
        L *= 2
    pos = current_k % original_len
    char = S[pos]
    if flips % 2 == 1:
        char = char.swapcase()
    result.append(char)

print(' '.join(result))