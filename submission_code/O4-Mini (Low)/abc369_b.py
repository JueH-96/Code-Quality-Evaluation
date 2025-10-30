def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    events = [input().split() for _ in range(N)]
    # Convert A_i to int, S_i is 'L' or 'R'
    events = [(int(a), s) for a, s in events]

    # dp maps (left_pos, right_pos) -> min fatigue so far
    # positions are 0..100, where 0 means "not yet placed / free start"
    dp = {(0, 0): 0}

    for key, hand in events:
        new_dp = {}
        if hand == 'L':
            # press with left hand: move left from l to key
            for (l, r), cost in dp.items():
                # movement cost: if l==0 (initial), cost 0, else abs(key - l)
                move_cost = 0 if l == 0 else abs(key - l)
                new_state = (key, r)
                new_cost = cost + move_cost
                prev = new_dp.get(new_state)
                if prev is None or new_cost < prev:
                    new_dp[new_state] = new_cost
        else:  # hand == 'R'
            # press with right hand
            for (l, r), cost in dp.items():
                move_cost = 0 if r == 0 else abs(key - r)
                new_state = (l, key)
                new_cost = cost + move_cost
                prev = new_dp.get(new_state)
                if prev is None or new_cost < prev:
                    new_dp[new_state] = new_cost

        dp = new_dp

    # answer is minimal cost among all end states
    ans = min(dp.values())
    print(ans)

if __name__ == "__main__":
    main()