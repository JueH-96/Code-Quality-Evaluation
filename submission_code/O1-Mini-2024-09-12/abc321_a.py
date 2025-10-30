N = input().strip()
if all(a > b for a, b in zip(N, N[1:])):
    print("Yes")
else:
    print("No")