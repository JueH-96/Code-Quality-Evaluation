def main():
    import sys
    sys.setrecursionlimit(10000)
    t = int(sys.stdin.readline())
    # dp[(mod3, moves_done, player)] = True/False (True means Vanya can force a win)
    from functools import lru_cache

    @lru_cache(None)
    def dfs(mod3, moves_done, player):
        # If we've already made 10 moves and never hit mod3==0 on Vanya's turn, Vanya loses
        if moves_done >= 10:
            return False

        # Vanya's turn: he wants ANY move that leads to a win
        if player == 0:
            for delta in (-1, 1):
                new_mod = (mod3 + delta) % 3
                new_moves = moves_done + 1
                # After Vanya's move, immediate win if divisible by 3
                if new_mod == 0:
                    return True
                # If that used up all 10 moves, it's a loss for Vanya
                if new_moves >= 10:
                    continue
                # Otherwise go to Vova's turn
                if dfs(new_mod, new_moves, 1):
                    return True
            return False

        # Vova's turn: he wants ALL Vanya responses to fail; if any branch makes Vanya lose, Vova will take it
        else:
            for delta in (-1, 1):
                new_mod = (mod3 + delta) % 3
                new_moves = moves_done + 1
                # If that uses up all moves, Vanya loses
                if new_moves >= 10:
                    return False
                # If in that branch Vanya cannot force a win, Vova will choose it
                if not dfs(new_mod, new_moves, 0):
                    return False
            # If every branch still lets Vanya force a win, Vova loses
            return True

    out = []
    for _ in range(t):
        n = int(sys.stdin.readline())
        m = n % 3
        if dfs(m, 0, 0):
            out.append("First")
        else:
            out.append("Second")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()