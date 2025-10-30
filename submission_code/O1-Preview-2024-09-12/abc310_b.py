N, M = map(int, input().split())
products = []
for _ in range(N):
    lst = input().split()
    P_i = int(lst[0])
    C_i = int(lst[1])
    F_i = set(map(int, lst[2:]))
    products.append((P_i, F_i))

found = False
for i in range(N):
    P_i, F_i = products[i]
    for j in range(N):
        if i == j:
            continue
        P_j, F_j = products[j]
        if P_i >= P_j and F_i.issubset(F_j):
            if P_i > P_j or (F_j - F_i):
                found = True
                break
    if found:
        break

print("Yes" if found else "No")