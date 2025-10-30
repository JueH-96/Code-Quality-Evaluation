# YOUR CODE HERE
N, L, R = map(int, input().split())
sequence = list(range(1, N+1))
sequence[L-1:R] = sequence[L-1:R][::-1]
print(' '.join(map(str, sequence)))