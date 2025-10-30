n = int(input())
L = []
R = []
for _ in range(n):
    l, r = map(int, input().split())
    L.append(l)
    R.append(r)

sum_L = sum(L)
sum_R = sum(R)

if sum_L > 0 or sum_R < 0:
    print("No")
else:
    X = L.copy()
    increase_needed = -sum_L
    
    for i in range(n):
        increase = min(increase_needed, R[i] - L[i])
        X[i] += increase
        increase_needed -= increase
    
    print("Yes")
    print(" ".join(map(str, X)))