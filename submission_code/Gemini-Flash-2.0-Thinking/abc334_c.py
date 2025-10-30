def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    remaining_socks = []
    counts = {}
    for i in range(1, n + 1):
        counts[i] = 2

    for color in a:
        counts[color] -= 1

    for color, count in counts.items():
        remaining_socks.extend([color] * count)

    remaining_socks.sort()

    num_remaining = len(remaining_socks)
    num_pairs = num_remaining // 2

    total_weirdness = 0
    for i in range(num_pairs):
        total_weirdness += abs(remaining_socks[2 * i + 1] - remaining_socks[2 * i])

    print(total_weirdness)

solve()