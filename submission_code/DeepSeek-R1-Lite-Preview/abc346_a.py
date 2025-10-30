N = int(input())
A = list(map(int, input().split()))
B = [a * b for a, b in zip(A, A[1:])]
print(' '.join(map(str, B)))