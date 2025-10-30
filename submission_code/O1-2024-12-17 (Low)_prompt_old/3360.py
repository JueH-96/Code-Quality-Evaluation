class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from bisect import bisect_left, bisect_right
        
        # Count frequencies of each character
        freq = [0]*26
        for ch in word:
            freq[ord(ch) - ord('a')] += 1
        
        # Filter out zero frequencies and sort
        f = sorted([x for x in freq if x > 0])
        if not f:  # if the string was empty or all deletions, cost 0
            return 0
        
        # Prefix sums for f; psum[i] = sum(f[:i])
        m = len(f)
        psum = [0]*(m+1)
        for i in range(m):
            psum[i+1] = psum[i] + f[i]
        
        max_f = f[-1]  # largest frequency
        min_cost = float('inf')
        
        # Sweep x from 0 to max_f (inclusive)
        # We pick final frequencies in the interval [x, x+k].
        for x in range(max_f + 1):
            # Number of elements strictly less than x
            left_idx = bisect_left(f, x)  
            # Number of elements <= x+k
            right_idx = bisect_right(f, x + k) - 1
            
            # Cost of removing frequencies that are < x
            # These must be removed completely (cost is just their sum)
            cost_left = psum[left_idx]
            
            # Cost of removing frequencies that are > x+k
            if right_idx < m - 1:
                # sum of all freq beyond right_idx
                sum_right = psum[m] - psum[right_idx+1]
                count_right = m - (right_idx + 1)
                # We reduce each of those freq[i] to (x+k)
                cost_right = sum_right - (x + k) * count_right
            else:
                # If right_idx == m-1, there are no frequencies > x+k
                cost_right = 0
            
            total_cost = cost_left + cost_right
            min_cost = min(min_cost, total_cost)
        
        return min_cost