# YOUR CODE HERE
N, L, R = map(int, input().split())
A = list(map(int, input().split()))

result = []
for a in A:
    if a < L:
        result.append(L)
    elif a > R:
        result.append(R)
    else:
        result.append(a)

print(' '.join(map(str, result)))