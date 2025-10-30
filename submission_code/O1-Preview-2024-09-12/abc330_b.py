# YOUR CODE HERE
N, L, R = map(int, input().split())
A = list(map(int, input().split()))
X = [min(max(a, L), R) for a in A]
print(' '.join(map(str, X)))