def main():
    import sys
    X, Y, Z = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    INF = 1 << 60
    previous = [INF] * 2
    previous[0] = 0  # initial state: Caps Lock off
    
    for c in S:
        current = [INF] * 2
        for s_prev in [0, 1]:
            if previous[s_prev] == INF:
                continue
            for k in [0, 1]:
                s_current = s_prev ^ k
                cost_so_far = previous[s_prev] + k * Z
                # Action a (cost X)
                if (s_current == 0 and c == 'a') or (s_current == 1 and c == 'A'):
                    if cost_so_far + X < current[s_current]:
                        current[s_current] = cost_so_far + X
                # Action Shift+a (cost Y)
                if (s_current == 0 and c == 'A') or (s_current == 1 and c == 'a'):
                    if cost_so_far + Y < current[s_current]:
                        current[s_current] = cost_so_far + Y
        previous = current
    
    print(min(previous))

if __name__ == "__main__":
    main()