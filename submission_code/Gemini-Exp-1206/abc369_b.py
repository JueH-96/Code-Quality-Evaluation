def solve():
    n = int(input())
    moves = []
    for _ in range(n):
        a, s = input().split()
        moves.append((int(a), s))

    min_fatigue = float('inf')

    for start_left in range(1, 101):
        for start_right in range(1, 101):
            fatigue = 0
            left_pos = start_left
            right_pos = start_right

            for key, hand in moves:
                if hand == 'L':
                    fatigue += abs(key - left_pos)
                    left_pos = key
                else:
                    fatigue += abs(key - right_pos)
                    right_pos = key

            min_fatigue = min(min_fatigue, fatigue)

    print(min_fatigue)

solve()