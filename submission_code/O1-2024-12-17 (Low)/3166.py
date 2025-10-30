class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count frequencies of each distinct value
        freq = Counter(nums)
        freq_vals = list(freq.values())
        n = len(nums)
        
        # A quick lower bound is the number of distinct values (each value needs at least one group).
        # An upper bound is n (each element in its own group).
        # We will binary-search for the smallest G in [low, high] that can form a valid assignment.
        low = len(freq_vals)
        high = n
        
        # Check feasibility of using G groups.
        # Let x = n // G, r = n % G. Then we must have exactly r groups of size (x+1) and (G-r) groups of size x.
        # Each frequency freq[v] can only be split into chunks of size x or x+1 (because all groups differ by at most 1).
        # That in turn implies for each freq[v], if we try to partition it into k groups, each chunk is either x or x+1.
        # A necessary condition to do this "homogeneously" is that for each freq[v],
        # freq[v] mod (x+1) must be either 0 or x.  (Because if leftover < x and leftover != 0,
        # it would create a chunk smaller than x, causing a difference > 1 with an x or x+1 group.)
        #
        # If leftover == x, we need 1 chunk of size x, and freq[v]//(x+1) chunks of size (x+1).
        # If leftover == 0, we only need freq[v]//(x+1) chunks of size (x+1) (possibly zero if freq[v]<x+1).
        #
        # Summing over all v, let sum_xplus1 = total number of (x+1)-chunks, sum_x = total number of x-chunks.
        # We must ensure sum_xplus1 <= r and sum_x <= G-r, else not feasible.
        def can_do(G: int) -> bool:
            # x and r describing how many groups are of size (x+1) or x
            x = n // G
            r = n % G
            sum_xplus1 = 0  # how many (x+1)-sized chunks in total
            sum_x_ = 0      # how many x-sized chunks in total
            
            # Edge case: if x == 0, then r == n (because G > n),
            # meaning we have n groups of size 1 and G-n groups of size 0.
            # Then each freq[v] must be <= 1 to fit into a 1-sized group alone.
            # (Because we can't combine different values in a single group.)
            if x == 0:
                # Then we have r = n groups of size 1, and (G-n) of size 0.
                # So we can assign each of the n elements to one of the size-1 groups
                # if and only if freq[v] <= 1 for all v.
                # Also we must have G >= number_of_distinct_values (already guaranteed by the search range).
                return all(fv <= 1 for fv in freq_vals)
            
            # For each frequency, check whether it can be split into chunks of size x or x+1
            for fv in freq_vals:
                big = x + 1
                # The remainder when dividing by (x+1)
                leftover = fv % big
                # If leftover != 0 or x, we can't form exact chunks of x or x+1
                # e.g. if leftover = 2 and x=3, big=4 => chunk of size 2 would differ from 3 or 4 by > 1.
                if leftover not in (0, x):
                    return False
                
                # How many full big-chunks (x+1) do we use?
                num_big_chunks = fv // big
                sum_xplus1 += num_big_chunks
                
                # If leftover == x, we need one x-chunk as well
                if leftover == x and leftover != 0:
                    sum_x_ += 1
                
                # If at any point we exceed the allowed counts of x or x+1 sized groups, fail early
                if sum_xplus1 > r or sum_x_ > (G - r):
                    return False
            
            # After checking all frequencies, we must not exceed the capacity of x+1 or x groups
            return (sum_xplus1 <= r) and (sum_x_ <= (G - r))
        
        # Binary-search over the range [low, high]
        ans = high
        while low <= high:
            mid = (low + high) // 2
            if can_do(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return ans