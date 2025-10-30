n = input().strip()
if len(n) == 1:
    print("Yes")
else:
    for i in range(len(n) - 1):
        if n[i] <= n[i + 1]:
            print("No")
            exit()
    print("Yes")