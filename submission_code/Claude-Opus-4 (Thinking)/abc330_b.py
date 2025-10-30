# YOUR CODE HERE
N, L, R = map(int, input().split())
A = list(map(int, input().split()))

X = [max(L, min(a, R)) for a in A]
print(" ".join(map(str, X)))