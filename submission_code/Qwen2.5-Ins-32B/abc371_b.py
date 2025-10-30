import sys

N, M = map(int, input().split())
families = [None] * (N + 1)  # Using 1-based indexing for families

for _ in range(M):
    A, B = input().split()
    A = int(A)
    if families[A] is None and B == 'M':
        families[A] = 'Taro'
        print('Yes')
    else:
        print('No')