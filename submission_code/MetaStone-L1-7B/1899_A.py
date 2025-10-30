t = int(input())
for _ in range(t):
    n = int(input())
    mod = n % 3
    if mod == 1 or mod == 2:
        print("First")
    else:
        print("Second")