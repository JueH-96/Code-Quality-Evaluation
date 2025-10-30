def main():
    import sys
    X, Y, Z = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    INF = 1 << 60
    cost_off = 0
    cost_on = INF
    for c in S:
        new_off = INF
        new_on = INF
        for prev_state in [0, 1]:
            prev_cost = cost_off if prev_state == 0 else cost_on
            # Option 1: action1 (X)
            char = 'a' if prev_state == 0 else 'A'
            if char == c:
                if prev_state == 0:
                    new_off = min(new_off, prev_cost + X)
                else:
                    new_on = min(new_on, prev_cost + X)
            # Option 2: action2 (Y)
            char = 'A' if prev_state == 0 else 'a'
            if char == c:
                if prev_state == 0:
                    new_off = min(new_off, prev_cost + Y)
                else:
                    new_on = min(new_on, prev_cost + Y)
            # Option 3: toggle + action1 (Z + X)
            new_state = 1 - prev_state
            char = 'a' if new_state == 0 else 'A'
            if char == c:
                total = prev_cost + Z + X
                if new_state == 0:
                    new_off = min(new_off, total)
                else:
                    new_on = min(new_on, total)
            # Option 4: toggle + action2 (Z + Y)
            new_state = 1 - prev_state
            char = 'A' if new_state == 0 else 'a'
            if char == c:
                total = prev_cost + Z + Y
                if new_state == 0:
                    new_off = min(new_off, total)
                else:
                    new_on = min(new_on, total)
        cost_off, cost_on = new_off, new_on
    print(min(cost_off, cost_on))

if __name__ == "__main__":
    main()