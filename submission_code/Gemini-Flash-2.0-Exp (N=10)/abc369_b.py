def solve():
    n = int(input())
    moves = []
    for _ in range(n):
        a, s = input().split()
        moves.append((int(a), s))

    min_fatigue = float('inf')

    for start_left in range(1, 101):
        for start_right in range(1, 101):
            current_fatigue = 0
            left_pos = start_left
            right_pos = start_right

            for a, s in moves:
                if s == 'L':
                    current_fatigue += abs(a - left_pos)
                    left_pos = a
                else:
                    current_fatigue += abs(a - right_pos)
                    right_pos = a
            
            min_fatigue = min(min_fatigue, current_fatigue)

    print(min_fatigue)

solve()