class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        """
        We want to count the number of ways to split nums into three contiguous subarrays:
          nums1 = nums[0:a], nums2 = nums[a:b], nums3 = nums[b:n]
        such that either
          (1) nums1 is a prefix of nums2  OR  (2) nums2 is a prefix of nums3.
        
        We will use rolling-hash to check subarray equality in O(1) after O(n) preprocessing.
        Then we count:
          • count1 = number of splits satisfying condition (1)
          • count2 = number of splits satisfying condition (2)
          • countBoth = number of splits satisfying both conditions
        The answer = count1 + count2 - countBoth  (union of the two conditions).
        
        ------------------------------------------------
        Condition (1): nums1 is prefix of nums2
          => length(nums1) <= length(nums2), i.e. a <= (b-a) => b >= 2a,
             subarray [0..a) == [a..2a).
          For each a, if [0..a) == [a..2a), then every b in [2a..n-1] is valid.
          So # of valid b's is (n-1) - (2a) + 1 = n - 2a.
          Summation over a = 1..floor(n/2).

        ------------------------------------------------
        Condition (2): nums2 is prefix of nums3
          => length(nums2) <= length(nums3), i.e. (b-a) <= (n-b) => 2b <= n + a,
             subarray [a..b) == [b..b+(b-a)].
          We check all valid (a,b) in O(n^2) using rolling hash (if each check is O(1)).

        ------------------------------------------------
        Intersection (both conditions):
          We again loop over a where [0..a) == [a..2a), then for b in [2a.. min(n-1, (n+a)//2]],
          check if [a..b) == [b..b+(b-a)]. That also is O(n^2) in worst case.

        Since n <= 5000, a careful O(n^2) solution with efficient hashing can pass in Python.
        ------------------------------------------------
        """
        import sys
        sys.setrecursionlimit(10**7)
        
        n = len(nums)
        if n < 3:
            return 0
        
        # Rolling-hash preparation
        # We'll use a single modulus; for safety we can choose a large prime
        # and a base. Risk of collisions is small for n up to 5000, but it can work in practice.
        base = 257
        mod = 10**9+7
        
        # Precompute powers of base
        pow_base = [1]*(n+1)
        for i in range(n):
            pow_base[i+1] = (pow_base[i] * base) % mod
        
        # Prefix hash array: H[i+1] = hash of nums[0..i] (length i+1)
        H = [0]*(n+1)
        for i in range(n):
            H[i+1] = (H[i]*base + nums[i]) % mod
        
        # Function to get hash of subarray nums[L..R-1] in O(1)
        def get_hash(L, R):
            # R is exclusive
            length = R - L
            return (H[R] - (H[L] * pow_base[length] % mod) + mod) % mod
        
        # Function to check if subarray [l1..r1) == [l2..r2) by length and hash
        def equal_subarrays(l1, r1, l2, r2):
            if (r1 - l1) != (r2 - l2):
                return False
            return get_hash(l1, r1) == get_hash(l2, r2)
        
        # ------------------------------------------------
        # count1: number of ways that satisfy (nums1 is prefix of nums2).
        # We only need to check whether [0..a) == [a..2a), then all b in [2a..n) except we must have b < n.
        # So that is (n-2a) possible b's if 2a <= n-1.
        
        matchLen = [False]*(n+1)  # matchLen[a] = True if [0..a) == [a..2a)
        max_a = n//2
        for a in range(1, max_a+1):  # a up to n//2
            # check subarray [0..a) ?= [a..2a)
            if equal_subarrays(0, a, a, 2*a):
                matchLen[a] = True
        
        count1 = 0
        for a in range(1, max_a+1):
            if matchLen[a]:
                # valid b in [2a..n-1]
                if 2*a <= n-1:
                    count1 += (n - 2*a)  # number of valid b's
        
        # ------------------------------------------------
        # count2: number of ways that satisfy (nums2 is prefix of nums3).
        # We do an O(n^2) check: for each b in [1..n-1], 
        #   for a in [max(0,2b - n)..b-1],
        #     let length = b-a; check if length <= n-b and subarray [a..b) = [b..b+length].
        
        count2 = 0
        for b in range(1, n):  # middle boundary
            # length(nums2) = b-a <= n-b => 2b <= n + a => a >= 2b - n
            start_a = max(0, 2*b - n)
            for a in range(start_a, b):
                length_sub2 = b - a
                if length_sub2 <= n - b:  # subarray2 fits as prefix in nums3
                    if equal_subarrays(a, b, b, b+length_sub2):
                        count2 += 1
        
        # ------------------------------------------------
        # countBoth: number of ways that satisfy both conditions.
        # That means:
        #   1) matchLen[a] is True (so [0..a) == [a..2a]) and b >= 2a
        #   2) b in [2a..n-1], plus (b-a) <= (n-b), i.e. b <= (n+a)//2
        # and subarray [a..b) == [b..b+(b-a)].
        
        countBoth = 0
        for a in range(1, max_a+1):
            if matchLen[a]:
                # b in [2a.. min(n-1, floor((n+a)//2))]
                b_low = 2*a
                b_high = min(n-1, (n+a)//2)
                for b in range(b_low, b_high+1):
                    length_sub2 = b - a
                    if length_sub2 <= n - b:  # needed for prefix check
                        if equal_subarrays(a, b, b, b+length_sub2):
                            countBoth += 1
        
        # Final answer by inclusion-exclusion of the two conditions
        return count1 + count2 - countBoth