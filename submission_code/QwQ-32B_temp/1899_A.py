import sys
from functools import lru_cache

def main():
    input = sys.stdin.read().split()
    t = int(input[0])
    cases = list(map(int, input[1:t+1]))
    
    for n in cases:
        @lru_cache(maxsize=None)
        def can_win(current_n, move_count):
            if move_count == 10:
                return False  # Vova wins, current player can't
            current_player_is_vanya = (move_count % 2 == 0)
            if current_player_is_vanya:
                # Vanya's turn: check all options
                for delta in (1, -1):
                    new_n = current_n + delta
                    if new_n % 3 == 0:
                        return True
                    # Check if next state is a losing position for opponent
                    if not can_win(new_n, move_count + 1):
                        return True
                return False
            else:
                # Vova's turn: check if any move leads to Vanya's loss
                for delta in (1, -1):
                    new_n = current_n + delta
                    if not can_win(new_n, move_count + 1):
                        return True
                return False
        
        result = can_win(n, 0)
        print("First" if result else "Second")

if __name__ == "__main__":
    main()