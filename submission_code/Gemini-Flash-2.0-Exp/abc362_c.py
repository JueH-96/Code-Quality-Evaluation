def solve():
    n = int(input())
    ranges = []
    for _ in range(n):
        l, r = map(int, input().split())
        ranges.append((l, r))

    min_sum = sum(l for l, r in ranges)
    max_sum = sum(r for l, r in ranges)

    if min_sum > 0 or max_sum < 0:
        print("No")
        return

    x = []
    current_sum = 0
    for i in range(n):
        l, r = ranges[i]
        
        val = max(l, min(r, -current_sum))
        x.append(val)
        current_sum += val

    if sum(x) == 0:
        print("Yes")
        print(*x)
    else:
        print("No")

solve()