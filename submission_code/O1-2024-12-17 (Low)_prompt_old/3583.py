class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        import sys
        import bisect
        
        MAX_VAL = 50000
        
        # 1) Count frequencies of each number in nums
        freq = [0]*(MAX_VAL+1)
        for num in nums:
            freq[num] += 1
        
        # 2) We will calculate how many pairs have gcd == d for each d in [1..MAX_VAL]
        #    Using a "multiples" approach from largest to smallest:
        #    Let countMultiples(d) = sum of freq[k*d] for k*d <= MAX_VAL.
        #    Then initially gcdCount[d] = comb(countMultiples(d), 2).
        #    But we then subtract gcdCount[m] for all m = 2*d, 3*d, ... to get only pairs with gcd exactly d.
        gcdCount = [0]*(MAX_VAL+1)
        
        # Precompute how many numbers are multiples of d
        # Then compute gcdCount[d] = number of pairs with gcd exactly d
        for d in range(MAX_VAL, 0, -1):
            # sum of freq of multiples
            c = 0
            multiple = d
            while multiple <= MAX_VAL:
                c += freq[multiple]
                multiple += d
            
            # number of pairs if gcd is at least d
            pairs = c*(c-1)//2
            
            # subtract pairs accounted for by multiples of d (i.e. > d)
            multiple = 2*d
            while multiple <= MAX_VAL:
                pairs -= gcdCount[multiple]
                multiple += d
            
            gcdCount[d] = pairs
        
        # gcdCount[d] now holds the number of pairs that have gcd exactly d.
        # 3) Build a prefix sum array so prefix[d] tells how many pairs have gcd <= d.
        prefix = [0]*(MAX_VAL+1)
        running = 0
        for d in range(1, MAX_VAL+1):
            running += gcdCount[d]
            prefix[d] = running
        
        # 4) Answer each query using binary search:
        #    We want the gcdPairs[ query ] (0-based).
        #    So we find the smallest d such that prefix[d] > query.
        
        answer = []
        for q in queries:
            # Binary search in [1..MAX_VAL]
            lo, hi = 1, MAX_VAL
            while lo < hi:
                mid = (lo + hi) // 2
                if prefix[mid] > q:
                    hi = mid
                else:
                    lo = mid + 1
            answer.append(lo)
        
        return answer