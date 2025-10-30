# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

B = [A[i] * A[i + 1] for i in range(N - 1)]
print(' '.join(map(str, B)))