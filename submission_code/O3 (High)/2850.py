from functools import lru_cache

class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        """
        pieces:
            0 -> "AA"
            1 -> "BB"
            2 -> "AB"
        a run of three identical letters must never appear
        """
        # adjacency lists that keep the run-of-three rule satisfied
        NEXT = {
            -1: (0, 1, 2),   # start of the string – every type is possible
            0 : (1,      ),  # AA  -> BB
            1 : (0, 2   ),   # BB  -> AA or AB
            2 : (0, 2   )    # AB  -> AA or AB   (never BB)
        }

        @lru_cache(None)
        def dfs(a: int, b: int, c: int, last: int) -> int:
            """
            returns the maximum number of pieces that can still be added
            with 'a' AA left, 'b' BB left, 'c' AB left and the last
            already placed piece of type 'last'
            """
            best = 0
            for nxt in NEXT[last]:
                if nxt == 0 and a:
                    best = max(best, 1 + dfs(a - 1, b,     c,     0))
                elif nxt == 1 and b:
                    best = max(best, 1 + dfs(a,     b - 1, c,     1))
                elif nxt == 2 and c:
                    best = max(best, 1 + dfs(a,     b,     c - 1, 2))
            return best

        pieces_used = dfs(x, y, z, -1)          # -1  denotes  “no previous piece”
        return pieces_used * 2                  # every piece contributes two characters