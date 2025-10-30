V1, V2, V3 = map(int, input().split())

if V1 < V2 * 3 or V1 > V2 * 3 + 343 or V2 < V3 or V2 > V3 * 7 or V3 > 343:
    print("No")
else:
    print("Yes")
    print(0, 0, 0, 0, 0, V3, 0, V3, 0)