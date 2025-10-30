def solve():
    n, q = map(int, input().split())
    queries = [int(input()) for _ in range(q)]

    for k in queries:
        if k == 0 or k == n * n:
            print("Yes")
        elif n == 2:
            if k == 4:
                print("Yes")
            else:
                print("No")
        elif n == 3:
            if k == 9 or k == 0:
                print("Yes")
            else:
                print("No")

        elif k % (n-1) == 0 and k <= n*n - n:
            print("Yes")
        
        else:
            print("No")

solve()