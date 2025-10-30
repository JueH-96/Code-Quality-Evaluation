def solve():
    n, t, p = map(int, input().split())
    l = list(map(int, input().split()))

    def count_eligible(lengths, target):
        count = 0
        for length in lengths:
            if length >= target:
                count += 1
        return count

    initial_count = count_eligible(l, t)
    if initial_count >= p:
        print(0)
        return

    days = 0
    while True:
        days += 1
        current_lengths = [length + days for length in l]
        current_count = count_eligible(current_lengths, t)
        if current_count >= p:
            print(days)
            return

solve()