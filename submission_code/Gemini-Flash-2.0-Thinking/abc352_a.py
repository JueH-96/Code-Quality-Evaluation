n, x, y, z = map(int, input().split())

if x < y:
    # Inbound train
    if x < z < y:
        print("Yes")
    else:
        print("No")
elif x > y:
    # Outbound train
    if y < z < x:
        print("Yes")
    else:
        print("No")