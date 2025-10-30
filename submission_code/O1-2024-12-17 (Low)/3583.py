class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        import sys
        import bisect
        
        max_val = max(nums)
        
        # Step 1: Frequency array of size up to max_val
        freq = [0]*(max_val+1)
        for x in nums:
            freq[x] += 1
        
        # Step 2: For each d, calculate how many numbers in nums are multiples of d
        multiples = [0]*(max_val+1)
        for d in range(1, max_val+1):
            count = 0
            for multiple in range(d, max_val+1, d):
                count += freq[multiple]
            multiples[d] = count
        
        # Step 3: Count how many pairs have gcd exactly d
        pairCount = [0]*(max_val+1)
        for d in range(max_val, 0, -1):
            # First compute how many pairs have gcd divisible by d
            c = multiples[d]
            total_pairs = c*(c-1)//2  # Combination of c items in pairs
            # Subtract pairs that have gcd as multiples of d (i.e. 2d, 3d, ...),
            # because they are counted in total_pairs but belong to a larger gcd.
            k = 2*d
            while k <= max_val:
                total_pairs -= pairCount[k]
                k += d
            pairCount[d] = total_pairs
        
        # Step 4: Build prefix of pair counts in ascending gcd order
        # prefixCount[d] = total number of pairs with gcd <= d
        prefixCount = [0]*(max_val+1)
        running = 0
        for d in range(1, max_val+1):
            running += pairCount[d]
            prefixCount[d] = running
        
        # We will answer: for a query q (0-based index),
        # find smallest d such that prefixCount[d] > q.
        
        # Step 5: Process each query
        ans = []
        for q in queries:
            # Binary search for smallest d with prefixCount[d] > q
            lo, hi = 1, max_val
            while lo < hi:
                mid = (lo + hi) // 2
                if prefixCount[mid] > q:
                    hi = mid
                else:
                    lo = mid + 1
            ans.append(lo)
        
        return ans