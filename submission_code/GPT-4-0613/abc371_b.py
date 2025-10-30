N, M = map(int, input().split())
families = [0]*N
for _ in range(M):
    A, B = input().split()
    A = int(A)
    if B == 'M' and families[A-1] == 0:
        families[A-1] = 1
        print('Yes')
    else:
        print('No')