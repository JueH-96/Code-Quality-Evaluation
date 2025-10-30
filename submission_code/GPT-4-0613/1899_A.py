t = int(input().strip())
for _ in range(t):
    n = int(input().strip())
    if n % 3 == 0 or (n+1) % 3 == 0:
        print("First")
    else:
        print("Second")