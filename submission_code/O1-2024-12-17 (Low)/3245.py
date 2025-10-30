class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        import bisect
        
        len_a = len(a)
        len_b = len(b)
        n = len(s)
        
        # Gather all starting positions where 'a' occurs
        indices_a = []
        for i in range(n - len_a + 1):
            if s[i:i+len_a] == a:
                indices_a.append(i)
        
        # Gather all starting positions where 'b' occurs
        indices_b = []
        for j in range(n - len_b + 1):
            if s[j:j+len_b] == b:
                indices_b.append(j)
        
        # We will use binary search on indices_b to check if there's a j
        # with |i - j| <= k for each i in indices_a
        result = []
        
        for i in indices_a:
            # lower_bound where j >= i-k
            # (if i-k is less than 0, bisect_left will go to index 0)
            lower = bisect.bisect_left(indices_b, i - k)
            # Check if this position in indices_b is within i+k
            if lower < len(indices_b) and abs(indices_b[lower] - i) <= k:
                result.append(i)
            else:
                # We can also check the next index (lower - 1) if it exists,
                # but only if 'lower' > 0. That might capture a j just below i-k.
                # However, since bisect_left gives the first position >= i-k,
                # we should also check the position immediately before 'lower' if it exists.
                if lower - 1 >= 0 and abs(indices_b[lower - 1] - i) <= k:
                    result.append(i)
        
        return result