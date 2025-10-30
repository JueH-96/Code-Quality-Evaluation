import sys
sys.setrecursionlimit(10000)

def main():
    data = sys.stdin.read().strip().split()
    # Read the 3x3 A grid into a flat list of length 9
    A = list(map(int, data))
    
    # All lines of length 3 on a 3x3: rows, columns, diagonals
    lines = [
        (0,1,2), (3,4,5), (6,7,8),  # rows
        (0,3,6), (1,4,7), (2,5,8),  # cols
        (0,4,8), (2,4,6)            # diags
    ]
    
    from functools import lru_cache
    
    @lru_cache(None)
    def dfs(pattern, turn):
        """
        pattern: a tuple of 9 ints in {0,1,2}, 0=white,1=red(T),2=blue(A)
        turn: 0 for Takahashi's move, 1 for Aoki's move
        Returns True if under optimal play from this state Takahashi wins,
        False if Aoki wins.
        """
        # 1) Check for an existing 3-in-a-row of one color
        for (i,j,k) in lines:
            c = pattern[i]
            if c != 0 and c == pattern[j] and c == pattern[k]:
                # if color is red(1), T has just made that and wins
                return (c == 1)
        
        # 2) If no whites left, compare scores
        if 0 not in pattern:
            sum_red  = 0
            sum_blue = 0
            for idx, col in enumerate(pattern):
                if col == 1:
                    sum_red += A[idx]
                elif col == 2:
                    sum_blue += A[idx]
            return (sum_red > sum_blue)
        
        # 3) Otherwise, try all moves for the current player
        if turn == 0:
            # Takahashi's turn: he paints a white cell red (1)
            # If any move leads to a state where dfs(...) is True,
            # he can force a win.
            for i, col in enumerate(pattern):
                if col == 0:
                    newp = list(pattern)
                    newp[i] = 1
                    if dfs(tuple(newp), 1):
                        return True
            return False
        else:
            # Aoki's turn: he paints a white cell blue (2)
            # He wants to find any move that makes dfs(...) False
            # (i.e., Aoki wins). If he can, he does so.
            for i, col in enumerate(pattern):
                if col == 0:
                    newp = list(pattern)
                    newp[i] = 2
                    if not dfs(tuple(newp), 0):
                        return False
            # If all moves still lead to True (Takahashi win),
            # then Takahashi wins from here.
            return True

    # Start from empty pattern, Takahashi to move first
    start_pattern = (0,)*9
    takahashi_wins = dfs(start_pattern, 0)
    
    print("Takahashi" if takahashi_wins else "Aoki")

if __name__ == "__main__":
    main()