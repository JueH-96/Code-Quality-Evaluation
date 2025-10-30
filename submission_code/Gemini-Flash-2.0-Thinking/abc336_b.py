def solve():
    n = int(input())
    count = 0
    if n % 2 != 0:
        print(0)
        return
    while n > 0 and n % 2 == 0:
        count += 1
        n //= 2
    print(count)

solve()