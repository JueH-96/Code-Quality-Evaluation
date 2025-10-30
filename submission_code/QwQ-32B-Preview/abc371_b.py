N, M = map(int, input().split())
has_eldest_son = [False] * (N + 1)

for _ in range(M):
    A, B = input().split()
    A = int(A)
    if B == 'M' and not has_eldest_son[A]:
        print('Yes')
        has_eldest_son[A] = True
    else:
        print('No')