# YOUR CODE HERE
N, T, P = map(int, input().split())
L = list(map(int, input().split()))
D = sorted([max(0, T - l) for l in L])
print(D[P - 1])