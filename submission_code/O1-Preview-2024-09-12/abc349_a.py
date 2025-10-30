# YOUR CODE HERE
N, *rest = map(int, open(0).read().split())
A = rest[:N-1]
print(-sum(A))