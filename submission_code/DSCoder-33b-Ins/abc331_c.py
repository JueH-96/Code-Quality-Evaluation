# YOUR CODE HERE

N = int(input())
A = list(map(int, input().split()))

total = sum(A)
B = [total - A[i] for i in range(N)]

print(' '.join(map(str, B)))