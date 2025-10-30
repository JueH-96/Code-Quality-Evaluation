def solve():
    n = int(input())
    a = list(map(int, input().split()))

    if n <= 1:
        print(0)
        return

    s = sum(a)
    floor_avg = s // n
    ceil_avg = (s + n - 1) // n

    ans = float('inf')

    # Target values: floor_avg and floor_avg + 1
    target_lower = floor_avg
    target_upper = floor_avg + 1

    increase_ops = 0
    decrease_ops = 0
    for x in a:
        if x < target_lower:
            increase_ops += target_lower - x
        elif x > target_upper:
            decrease_ops += x - target_upper
        elif x == target_lower:
            pass
        elif x == target_upper:
            pass

    ans = min(ans, max(increase_ops, decrease_ops))

    print(ans)

solve()