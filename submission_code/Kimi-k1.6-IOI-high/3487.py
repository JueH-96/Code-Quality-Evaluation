class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        t_set = set(targetIndices)
        is_target = [i in t_set for i in range(n)]
        
        # Initialize DP array where previous[j] is the minimal cost for the first j characters in source
        previous = [0] * (n + 1)
        
        for i in range(1, m + 1):
            current = [float('inf')] * (n + 1)
            pat_char = pattern[i-1]
            for j in range(1, n + 1):
                src_char = source[j-1]
                if src_char == pat_char:
                    # Option 1: take this character, add cost if it's a target index
                    cost = 1 if is_target[j-1] else 0
                    take = previous[j-1] + cost
                    # Option 2: don't take this character
                    not_take = current[j-1]
                    current[j] = min(take, not_take)
                else:
                    # Can't take this character, carry over the previous value
                    current[j] = current[j-1]
            previous = current
        
        minimal_cost = previous[n]
        return len(targetIndices) - minimal_cost if minimal_cost != float('inf') else 0