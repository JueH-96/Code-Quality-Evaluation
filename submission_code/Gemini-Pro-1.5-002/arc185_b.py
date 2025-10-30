def solve():
    n = int(input())
    a = list(map(int, input().split()))

    total_sum = sum(a)

    if total_sum % n != 0:
        print("No")
        return

    target = total_sum // n

    for x in a:
        if x > target:
            print("No")
            return
    
    print("Yes")


t = int(input())
for _ in range(t):
    solve()