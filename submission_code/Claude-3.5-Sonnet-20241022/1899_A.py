t = int(input())
for _ in range(t):
    n = int(input())
    
    # Try each possible move sequence up to 10 moves
    def can_win(num, moves):
        if moves > 10:
            return False
        if moves % 2 == 1:  # Vanya's move
            if (num + 1) % 3 == 0 or (num - 1) % 3 == 0:
                return True
            return can_win(num + 1, moves + 1) or can_win(num - 1, moves + 1)
        else:  # Vova's move
            if (num + 1) % 3 == 0 and (num - 1) % 3 == 0:
                return False
            if (num + 1) % 3 == 0:
                return can_win(num - 1, moves + 1)
            if (num - 1) % 3 == 0:
                return can_win(num + 1, moves + 1)
            return can_win(num + 1, moves + 1) and can_win(num - 1, moves + 1)
    
    if can_win(n, 1):
        print("First")
    else:
        print("Second")