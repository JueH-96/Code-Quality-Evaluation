class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # We label the block types as follows:
        # A = "AA", B = "BB", C = "AB"
        # We'll use integers to represent these states in our DP/DFS:
        #   0 => "no block chosen yet" (start)
        #   1 => A-block ("AA")
        #   2 => B-block ("BB")
        #   3 => C-block ("AB")
        #
        # Adjacency is determined by whether concatenating blockX + blockY
        # creates "AAA" or "BBB". From analysis, the allowed transitions are:
        #   0 -> 1,2,3  (starting with any block)
        #   1 -> 2      ("AA" -> "BB" => "AABB" is safe)
        #   2 -> 1,3    ("BB" -> "AA" => "BBAA", "BB" -> "AB" => "BBAB" are safe)
        #   3 -> 1,3    ("AB" -> "AA" => "ABAA", "AB" -> "AB" => "ABAB" are safe)
        #
        # We want the longest sequence of blocks (each length 2) subject to
        # using at most x A-blocks, y B-blocks, z C-blocks, and respecting
        # the above adjacency.
        #
        # We'll use DFS with memoization. State = (a_rem, b_rem, c_rem, last_block).
        # Returns the maximum number of blocks we can form from that state.

        from functools import lru_cache

        # Adjacency list for the "last block" -> possible next blocks
        adj = {
            0: [1, 2, 3],  # start: can choose any block
            1: [2],       # A -> B
            2: [1, 3],    # B -> A or C
            3: [1, 3],    # C -> A or C
        }

        @lru_cache(None)
        def dfs(a, b, c, last):
            best = 0
            for nxt in adj[last]:
                if nxt == 1 and a > 0:
                    best = max(best, 1 + dfs(a - 1, b, c, 1))
                elif nxt == 2 and b > 0:
                    best = max(best, 1 + dfs(a, b - 1, c, 2))
                elif nxt == 3 and c > 0:
                    best = max(best, 1 + dfs(a, b, c - 1, 3))
            return best

        # The DFS returns the maximum count of blocks; each block is length 2.
        return 2 * dfs(x, y, z, 0)