def solve():
    n = int(input())
    actions = []
    for _ in range(n):
        a, s = input().split()
        actions.append((int(a), s))

    dp = {}  # (index, left_pos, right_pos) -> min_fatigue

    def get_dp(i, l, r):
        return dp.get((i, l, r), float('inf'))

    dp[(0, 0, 0)] = 0  # Initial state, 0 fatigue

    for i in range(1, n + 1):
        a, s = actions[i - 1]
        new_dp = {}
        for (prev_i, left, right), fatigue in dp.items():
            if prev_i == i - 1:
                if s == 'L':
                    new_fatigue = fatigue + abs(left - a)
                    new_dp[(i, a, right)] = min(new_dp.get((i, a, right), float('inf')), new_fatigue)
                else:
                    new_fatigue = fatigue + abs(right - a)
                    new_dp[(i, left, a)] = min(new_dp.get((i, left, a), float('inf')), new_fatigue)
        dp.update(new_dp)

    min_fatigue = float('inf')
    for (i, l, r), fatigue in dp.items():
        if i == n:
            min_fatigue = min(min_fatigue, fatigue)

    print(min_fatigue)

solve()