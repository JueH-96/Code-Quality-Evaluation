def solve():
    n, q = map(int, input().split())
    instructions = [input().split() for _ in range(q)]

    left = 1
    right = 2
    total_moves = 0

    def distance(a, b):
        diff = abs(a - b)
        return min(diff, n - diff)

    def move(current, target, avoid):
        if current == target:
            return 0

        dist1 = (target - current + n) % n
        dist2 = (current - target + n) % n

        if dist1 <= dist2:
            path = [(current + i - 1) % n + 1 for i in range(1, dist1 + 1)]
            if avoid not in path[:-1]:
                return dist1
        if dist2 <= dist1:
            path = [(current - i - 1 + n) % n + 1 for i in range(1, dist2 + 1)]
            if avoid not in path[:-1]:
                return dist2
        return float('inf')

    for h, t_str in instructions:
        target = int(t_str)

        if h == 'L':
            if left != target:
                direct_moves = move(left, target, right)
                if direct_moves != float('inf'):
                    total_moves += direct_moves
                    left = target
                else:
                    # Move right hand
                    moves1 = 1 + move(left, target, (right % n) + 1 if right < n else 1)
                    moves2 = 1 + move(left, target, (right - 2 + n) % n + 1)

                    if moves1 <= moves2:
                        total_moves += 1
                        right = (right % n) + 1 if right < n else 1
                        total_moves += move(left, target, right)
                        left = target
                    else:
                        total_moves += 1
                        right = (right - 2 + n) % n + 1
                        total_moves += move(left, target, right)
                        left = target

        elif h == 'R':
            if right != target:
                direct_moves = move(right, target, left)
                if direct_moves != float('inf'):
                    total_moves += direct_moves
                    right = target
                else:
                    moves1 = 1 + move(right, target, (left % n) + 1 if left < n else 1)
                    moves2 = 1 + move(right, target, (left - 2 + n) % n + 1)

                    if moves1 <= moves2:
                        total_moves += 1
                        left = (left % n) + 1 if left < n else 1
                        total_moves += move(right, target, left)
                        right = target
                    else:
                        total_moves += 1
                        left = (left - 2 + n) % n + 1
                        total_moves += move(right, target, left)
                        right = target

    print(total_moves)

solve()