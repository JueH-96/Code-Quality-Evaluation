class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        import bisect
        
        # Find all occurrences of pattern p in string s
        # Returns a list of start indices
        def findOccurrences(s, p):
            occurrences = []
            start_idx = 0
            p_len = len(p)
            while True:
                # find returns -1 if p not found
                start_idx = s.find(p, start_idx)
                if start_idx == -1:
                    break
                occurrences.append(start_idx)
                start_idx += 1  # move forward to find next occurrence
            return occurrences
        
        # Get all indices for 'a' and 'b'
        a_occurrences = findOccurrences(s, a)
        b_occurrences = findOccurrences(s, b)
        
        # Sort occurrences of b (they should already be sorted but we ensure it)
        b_occurrences.sort()
        
        # For each i in a_occurrences, we check if there's a j in b_occurrences
        # such that |i - j| <= k. We'll do this using binary search.
        res = []
        for i in a_occurrences:
            # We're looking for j in range [i - k, i + k]
            left_bound = i - k
            right_bound = i + k
            
            # Use bisect to find the position in b_occurrences where
            # left_bound could be inserted to keep sorted order
            left_idx = bisect.bisect_left(b_occurrences, left_bound)
            
            # Now, as long as b_occurrences[left_idx] is <= right_bound (and within array)
            # we have a valid j
            if left_idx < len(b_occurrences) and b_occurrences[left_idx] <= right_bound:
                res.append(i)
            else:
                # We might need to check if left_idx > 0 and left_idx - 1 is still in range
                # but the main check above should suffice if there's a valid j within [left_bound, right_bound].
                # However, let's handle edge case carefully:
                # If left_idx > 0, check b_occurrences[left_idx - 1] if it exists
                if left_idx > 0 and b_occurrences[left_idx - 1] >= left_bound and b_occurrences[left_idx - 1] <= right_bound:
                    res.append(i)
        
        return res