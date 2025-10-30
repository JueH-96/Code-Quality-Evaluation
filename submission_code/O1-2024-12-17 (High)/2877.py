class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        from itertools import permutations
        
        # Helper function to merge two strings into their shortest common superstring
        def merge2(x: str, y: str) -> str:
            # If one string is already fully contained in the other, return the larger container
            if y in x:
                return x
            if x in y:
                return y
            
            # Otherwise, find the largest overlap of x's suffix with y's prefix
            max_overlap = 0
            min_len = min(len(x), len(y))
            for length in range(min_len, 0, -1):
                if x.endswith(y[:length]):
                    max_overlap = length
                    break
            
            # Merge them based on the overlap
            return x + y[max_overlap:]
        
        # Try all permutations of (a, b, c) and keep track of the best answer
        best = None
        for perm in permutations([a, b, c]):
            merged_ab = merge2(perm[0], perm[1])
            merged_abc = merge2(merged_ab, perm[2])
            if best is None:
                best = merged_abc
            else:
                # If we find a shorter or lexicographically smaller string, update best
                if len(merged_abc) < len(best) or (len(merged_abc) == len(best) and merged_abc < best):
                    best = merged_abc
                    
        return best