# YOUR CODE HERE
S, T = input().split()
n, m = len(S), len(T)
for c in range(1, n):
    for w in range(c, n-1):
        if all(S[i+c-1] == T[i] for i in range(m) if i*(w+1)+c-1 < n):
            print('Yes')
            exit()
print('No')