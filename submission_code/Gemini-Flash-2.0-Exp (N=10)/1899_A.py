def solve():
    t = int(input())
    for _ in range(t):
        n = int(input())
        
        if n % 3 == 0:
            print("First")
        elif (n + 1) % 3 == 0:
            print("First")
        elif (n - 1) % 3 == 0:
            if (n + 2) % 3 == 0 or (n-2) % 3 == 0:
                print("First")
            else:
                print("Second")
        else:
            print("Second")

solve()