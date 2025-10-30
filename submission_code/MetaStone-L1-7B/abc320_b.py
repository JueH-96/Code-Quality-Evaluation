S = input().strip()
n = len(S)

for L in range(n, 0, -1):
    for i in range(n - L + 1):
        substr = S[i:i+L]
        if substr == substr[::-1]:
            print(L)
            exit()