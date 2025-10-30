def solve():
    n = int(input())
    a = list(map(int, input().split()))

    if n == 2:
        print("Yes")
        return

    ratio = a[1] / a[0]
    for i in range(2, n):
        if abs(a[i] / a[i-1] - ratio) > 1e-9:
            print("No")
            return
    
    print("Yes")

solve()