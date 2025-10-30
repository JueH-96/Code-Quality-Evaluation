S, T = input().strip().split()
n, m = len(S), len(T)

for w in range(1, n):
    for c in range(1, w+1):
        if ''.join([S[i*w+c-1] for i in range(n//w) if i*w+c-1<n]) == T:
            print('Yes')
            exit(0)

print('No')