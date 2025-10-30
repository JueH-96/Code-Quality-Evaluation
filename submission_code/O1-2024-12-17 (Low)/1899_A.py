def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    # We'll implement a minimax-style DFS with memo, 
    # tracking the (current_value, moves_done, whose_turn) state.
    # Return True if that position is winning for Vanya (the "First" player),
    # otherwise False.
    #
    # Game rules in short:
    #  - Turn order: Vanya (0) -> Vova (1) -> Vanya (0) -> ...
    #  - Each move: add or subtract 1.
    #  - If after Vanya's move the integer is divisible by 3, Vanya wins immediately.
    #  - If 10 moves total have passed and Vanya hasn't won, Vova wins.
    #
    # We'll define is_winning(n, move, turn) returning
    # True/False from Vanya's perspective (True = Vanya can force a win).
    
    from functools import lru_cache

    @lru_cache(None)
    def is_winning(value, move, turn):
        # If we've used all 10 moves, Vanya hasn't won => Vova wins => return False.
        if move == 10:
            return False
        
        if turn == 0:
            # Vanya's turn.
            # He can do value-1 or value+1.
            # If either move leads to an immediate multiple of 3 => immediate win.
            # Otherwise, the game continues. 
            # If the resulting next state is losing for the opponent => current is winning.
            candidates = []
            for next_val in (value - 1, value + 1):
                if next_val % 3 == 0:
                    # Immediate Vanya win
                    return True
                # Otherwise, see if that leads to a losing position for Vova
                candidates.append(is_winning(next_val, move + 1, 1))
            # If any of these is False => that means we found a move that leaves Vova in a losing position => Vanya wins
            return not all(candidates)
        else:
            # Vova's turn.
            # He can do value-1 or value+1, no immediate winning condition for Vova,
            # but if Vova can force a position that is losing for Vanya => current is losing for Vanya.
            # If both next states are winning for Vanya => current is winning for Vanya.
            next1 = is_winning(value - 1, move + 1, 0)
            next2 = is_winning(value + 1, move + 1, 0)
            # If Vova can pick a move leading to False for Vanya => we return False immediately
            if (not next1) or (not next2):
                return False
            else:
                return True

    idx = 1
    results = []
    for _ in range(t):
        n = int(input_data[idx]); idx += 1
        # Clear cache for each test to avoid collisions (or we can just not clear it because states won't overlap much).
        # But safer to clear:
        is_winning.cache_clear()
        # Start at move=0, turn=0 (Vanya).
        # The integer is currently n, and it's Vanya's turn to move.
        # If is_winning(n, 0, 0) is True => "First", else "Second"
        if is_winning(n, 0, 0):
            results.append("First")
        else:
            results.append("Second")
    
    print("
".join(results))

# Do not forget to call main
main()