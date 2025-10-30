def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    if n <= 1:
        print("Yes")
        return
    
    first = a[0]
    for i in range(1, n):
        if a[i] != first:
            print("No")
            return
    
    print("Yes")

solve()