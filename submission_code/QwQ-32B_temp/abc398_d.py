def main():
    import sys
    N, R, C = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    dx_prev = 0
    dy_prev = 0
    current_set = set()
    current_set.add((0, 0))
    result = []

    for c in S:
        dr, dc = 0, 0
        if c == 'N':
            dr = -1
        elif c == 'W':
            dc = -1
        elif c == 'S':
            dr = 1
        elif c == 'E':
            dc = 1
        else:
            assert False, "Invalid direction"

        dx_current = dx_prev + dr
        dy_current = dy_prev + dc

        key = (dx_current, dy_current)
        if key not in current_set:
            current_set.add(key)

        target_x = dx_current - R
        target_y = dy_current - C
        target = (target_x, target_y)

        if target in current_set:
            result.append('1')
        else:
            result.append('0')

        dx_prev, dy_prev = dx_current, dy_current

    print(''.join(result))

if __name__ == "__main__":
    main()