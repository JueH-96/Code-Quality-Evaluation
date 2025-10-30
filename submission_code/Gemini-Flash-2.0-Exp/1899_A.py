def solve():
    n = int(input())
    
    if n % 3 == 0:
        print("Second")
    elif (n + 1) % 3 == 0:
        print("First")
    elif (n - 1) % 3 == 0:
        print("First")
    else:
        if n % 3 == 1:
            print("First")
        else:
            print("First")

t = int(input())
for _ in range(t):
    solve()