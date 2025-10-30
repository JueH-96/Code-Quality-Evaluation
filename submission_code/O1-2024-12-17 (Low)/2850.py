class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        """
        We have:
          - x blocks of "AA"
          - y blocks of "BB"
          - z blocks of "AB"
        We want to concatenate a selection (possibly all or none) of these blocks,
        in some order, to form the longest string that has no "AAA" or "BBB" substring.
        
        Observing allowable transitions (where a block can follow another without
        creating "AAA" or "BBB"):
            "AA" -> "BB"
            "BB" -> "AA", "AB"
            "AB" -> "AA", "AB"
        
        We use DFS + memoization to explore the maximal chain length. The string length
        contribution of each block is always 2. We keep track of the last placed block
        and move to a compatible next block, decreasing the available count of that block.
        """
        from functools import lru_cache

        # Possible transitions: block -> set of blocks that can follow it
        transitions = {
            "AA": ["BB"],
            "BB": ["AA", "AB"],
            "AB": ["AA", "AB"]
        }
        
        # We'll use DFS with memo to find the maximum total length.
        @lru_cache(None)
        def dfs(a_left, b_left, ab_left, last_block):
            best = 0
            # If last_block is None (start), we can pick any type if available.
            if last_block is None:
                # Try "AA"
                if a_left > 0:
                    best = max(best, 2 + dfs(a_left - 1, b_left, ab_left, "AA"))
                # Try "BB"
                if b_left > 0:
                    best = max(best, 2 + dfs(a_left, b_left - 1, ab_left, "BB"))
                # Try "AB"
                if ab_left > 0:
                    best = max(best, 2 + dfs(a_left, b_left, ab_left - 1, "AB"))
            else:
                # Follow transitions
                for nxt_block in transitions[last_block]:
                    if nxt_block == "AA" and a_left > 0:
                        best = max(best, 2 + dfs(a_left - 1, b_left, ab_left, "AA"))
                    elif nxt_block == "BB" and b_left > 0:
                        best = max(best, 2 + dfs(a_left, b_left - 1, ab_left, "BB"))
                    elif nxt_block == "AB" and ab_left > 0:
                        best = max(best, 2 + dfs(a_left, b_left, ab_left - 1, "AB"))
            
            return best
        
        return dfs(x, y, z, None)