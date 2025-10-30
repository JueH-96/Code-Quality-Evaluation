def solve():
    n = int(input())
    p = list(map(int, input().split()))

    max_rating = -float('inf')

    for i in range(1, 1 << n):
        chosen_indices = []
        for j in range(n):
            if (i >> j) & 1:
                chosen_indices.append(j)

        if not chosen_indices:
            continue

        chosen_performances = [p[idx] for idx in chosen_indices]
        k = len(chosen_performances)

        numerator = 0
        denominator = 0

        for idx, perf in enumerate(chosen_performances):
            numerator += (0.9)**(k - 1 - idx) * perf
            denominator += (0.9)**(k - 1 - idx)

        rating = numerator / denominator - 1200 / k**0.5
        max_rating = max(max_rating, rating)

    print(f"{max_rating:.10f}")

solve()