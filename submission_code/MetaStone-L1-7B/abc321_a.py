n = input().strip()
if len(n) == 1:
    print("Yes")
else:
    is_321 = True
    for i in range(len(n) - 1):
        if n[i] <= n[i+1]:
            is_321 = False
            break
    print("Yes" if is_321 else "No")