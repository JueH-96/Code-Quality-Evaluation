def solve():
    n = int(input())
    moves = []
    for _ in range(n):
        a, s = input().split()
        moves.append((int(a), s))

    min_fatigue = float('inf')

    for initial_left in range(1, 101):
        for initial_right in range(1, 101):
            fatigue = 0
            left_pos = initial_left
            right_pos = initial_right

            for a, s in moves:
                a = int(a)
                if s == 'L':
                    fatigue += abs(a - left_pos)
                    left_pos = a
                else:
                    fatigue += abs(a - right_pos)
                    right_pos = a

            min_fatigue = min(min_fatigue, fatigue)

    print(min_fatigue)

solve()