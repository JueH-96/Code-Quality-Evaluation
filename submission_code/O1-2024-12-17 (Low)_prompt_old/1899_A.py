def solve():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    # We'll do a DFS with memo to determine if Vanya ("First") can force a win.

    from functools import lru_cache

    @lru_cache(None)
    def can_vanya_win(n, move_index):
        """
        Returns True if Vanya can force a win starting with integer n
        at move_index (0-based, even = Vanya's turn, odd = Vova's turn),
        given that there are at most 10 moves in total.
        """
        # If we have used up 10 moves, Vanya failed to produce divisible by 3 -> Vova wins
        if move_index == 10:
            return False  # Vanya didn't make it in time

        if move_index % 2 == 0:
            # Vanya's turn
            # If Vanya can directly move to a multiple of 3, he wins immediately
            if (n + 1) % 3 == 0 or (n - 1) % 3 == 0:
                return True
            # Otherwise, he tries both +1 and -1 and hopes for a winning path
            return can_vanya_win(n + 1, move_index + 1) or can_vanya_win(n - 1, move_index + 1)
        else:
            # Vova's turn
            # If Vova can move to a position from which Vanya cannot force a win, he'll do it
            # That means if either (n+1) or (n-1) leads to a position losing for Vanya (False),
            # Vova will choose that and make Vanya lose. So from the current perspective,
            # that means can_vanya_win() is False if ANY child is False.
            move1 = can_vanya_win(n + 1, move_index + 1)
            move2 = can_vanya_win(n - 1, move_index + 1)
            # If any move leads to False for Vanya, Vova will pick it -> result is False for Vanya
            if not move1 or not move2:
                return False
            # Otherwise, if both moves lead to True, Vova can't avoid losing -> True for Vanya
            return True

    idx = 1
    for _ in range(t):
        n = int(data[idx]); idx += 1
        # Clear memo for each test case so states don't overlap
        can_vanya_win.cache_clear()
        # If can_vanya_win is True => "First", else "Second"
        if can_vanya_win(n, 0):
            print("First")
        else:
            print("Second")

# Let's call solve() to comply with the prompt
solve()