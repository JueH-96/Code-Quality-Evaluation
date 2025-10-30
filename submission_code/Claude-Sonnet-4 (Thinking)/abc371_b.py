N, M = map(int, input().split())
families_with_eldest_son = set()

for i in range(M):
    A, B = input().split()
    A = int(A)
    
    if B == 'M' and A not in families_with_eldest_son:
        print("Yes")
        families_with_eldest_son.add(A)
    else:
        print("No")