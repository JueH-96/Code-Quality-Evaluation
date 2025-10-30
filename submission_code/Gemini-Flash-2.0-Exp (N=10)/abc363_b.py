def solve():
    n, t, p = map(int, input().split())
    l = list(map(int, input().split()))

    def count_at_least_t(lengths, target_t):
        count = 0
        for length in lengths:
            if length >= target_t:
                count += 1
        return count

    current_count = count_at_least_t(l, t)
    if current_count >= p:
        print(0)
        return

    days = 0
    while True:
        days += 1
        new_lengths = [length + days for length in l]
        new_count = count_at_least_t(new_lengths, t)
        if new_count >= p:
            print(days)
            return

solve()