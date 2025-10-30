def solve():
    strings = [input().strip() for _ in range(12)]
    count = sum(1 for i, s in enumerate(strings, start=1) if len(s) == i)
    print(count)

solve()