def solve():
    n, q = map(int, input().split())
    instructions = [input().split() for _ in range(q)]

    left = 1
    right = 2
    total_moves = 0

    for h_str, t_str in instructions:
        target = int(t_str)

        if h_str == 'L':
            blocked = right

            dist_cw = 0
            current = left
            possible_cw = True
            while current != target:
                next_cw = (current % n) + 1
                if next_cw > n:
                    next_cw = 1
                if next_cw == blocked:
                    possible_cw = False
                    break
                current = next_cw
                dist_cw += 1

            dist_ccw = 0
            current = left
            possible_ccw = True
            while current != target:
                next_ccw = ((current - 2 + n) % n) + 1
                if next_ccw == blocked:
                    possible_ccw = False
                    break
                current = next_ccw
                dist_ccw += 1

            moves = float('inf')
            if possible_cw:
                moves = min(moves, dist_cw)
            if possible_ccw:
                moves = min(moves, dist_ccw)

            total_moves += moves
            left = target

        elif h_str == 'R':
            blocked = left

            dist_cw = 0
            current = right
            possible_cw = True
            while current != target:
                next_cw = (current % n) + 1
                if next_cw > n:
                    next_cw = 1
                if next_cw == blocked:
                    possible_cw = False
                    break
                current = next_cw
                dist_cw += 1

            dist_ccw = 0
            current = right
            possible_ccw = True
            while current != target:
                next_ccw = ((current - 2 + n) % n) + 1
                if next_ccw == blocked:
                    possible_ccw = False
                    break
                current = next_ccw
                dist_ccw += 1

            moves = float('inf')
            if possible_cw:
                moves = min(moves, dist_cw)
            if possible_ccw:
                moves = min(moves, dist_ccw)

            total_moves += moves
            right = target

    print(int(total_moves))

solve()