import sys

def main() -> None:
    # read input
    values = []
    for _ in range(3):
        values.extend(map(int, sys.stdin.readline().split()))

    # pre–compute every winning line (bit masks)
    lines = []
    # rows
    for r in range(3):
        mask = 0
        for c in range(3):
            mask |= 1 << (r * 3 + c)
        lines.append(mask)
    # columns
    for c in range(3):
        mask = 0
        for r in range(3):
            mask |= 1 << (r * 3 + c)
        lines.append(mask)
    # diagonals
    lines.append((1 << 0) | (1 << 4) | (1 << 8))
    lines.append((1 << 2) | (1 << 4) | (1 << 6))

    ALL_MASK = (1 << 9) - 1

    # pre–compute the sum of the numbers for every subset of cells
    subset_sum = [0] * (1 << 9)
    for mask in range(1 << 9):
        s = 0
        for i in range(9):
            if mask & (1 << i):
                s += values[i]
        subset_sum[mask] = s

    from functools import lru_cache

    # helper to check if someone has already formed a 3-in-a-row
    def winner(t_mask: int, a_mask: int) -> int:
        for m in lines:
            if (t_mask & m) == m:
                return 1      # Takahashi has a line
            if (a_mask & m) == m:
                return 2      # Aoki has a line
        return 0              # no line yet

    @lru_cache(maxsize=None)
    def solve(t_mask: int, a_mask: int) -> bool:
        """
        Returns True  – Takahashi wins with perfect play from this position.
        Returns False – Aoki     wins with perfect play from this position.
        The position is completely described by the two bit masks because the
        player to move is determined by the counts of stones:
           · equal counts   -> Takahashi to move
           · otherwise      -> Aoki      to move
        """

        w = winner(t_mask, a_mask)
        if w == 1:
            return True
        if w == 2:
            return False

        # board full: decide by total score
        if (t_mask | a_mask) == ALL_MASK:
            diff = subset_sum[t_mask] - subset_sum[a_mask]   # >0 => T wins
            return diff > 0

        takahashi_turn = (t_mask.bit_count() == a_mask.bit_count())
        empty = (~(t_mask | a_mask)) & ALL_MASK

        if takahashi_turn:                       # Takahashi tries to make result True
            while empty:
                bit = empty & -empty             # least significant empty square
                empty -= bit
                new_t_mask = t_mask | bit

                # immediate win?
                if any((new_t_mask & m) == m for m in lines):
                    return True

                if solve(new_t_mask, a_mask):    # he can force a win through this move
                    return True
            return False                         # all moves lose
        else:                                    # Aoki tries to make result False
            while empty:
                bit = empty & -empty
                empty -= bit
                new_a_mask = a_mask | bit

                # immediate win for Aoki
                if any((new_a_mask & m) == m for m in lines):
                    return False

                if not solve(t_mask, new_a_mask):    # Aoki can force T's loss
                    return False
            return True                              # Takahashi survives every reply

    print('Takahashi' if solve(0, 0) else 'Aoki')


if __name__ == '__main__':
    main()