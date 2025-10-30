class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        """
        We have an infinite repetition of nums and want the length of the shortest subarray whose sum is exactly target.
        If no such subarray exists, return -1.

        Key idea (all nums[i] > 0):
           Let s = sum(nums).  There are three main scenarios:
             1) If s == target, the whole array has the exact sum => answer = len(nums).
             2) If s >  target, we only need to check up to 2 copies of nums (two-pointer/sliding-window) to find sum=target.
             3) If s <  target, let k = target // s, r = target % s.
                - If r == 0, exactly k copies of nums sum to target => length = k * len(nums).
                  (No shorter subarray is possible because skipping any positive elements drops sum below k*s.)
                - Otherwise, target = k*s + r.  Possible ways:
                   A) Take k full arrays plus a subarray that sums to r in (up to 2 copies of nums).
                   B) Or take (k+1) full arrays (which sum to (k+1)*s) and remove a prefix/suffix of sum = (k+1)*s - target = s - r.
                      Only removing a prefix or a suffix keeps the remainder contiguous.
                
                We compute:
                  candidate1 = k*n + minSubarraySum(r)         # if a subarray of sum=r exists
                  candidate2 = min( (k+1)*n - prefixLen , (k+1)*n - suffixLen )
                                where prefixLen is smallest prefix of up to 2 copies with sum = s-r,
                                      suffixLen is smallest suffix of up to 2 copies with sum = s-r.
                The answer is the minimum valid among these candidates. If none is valid => -1.

        All helper searches are done in O(n) using two-pointer or prefix/suffix scans (because we only ever need at most 2 copies of nums).
        """
        import sys
        sys.setrecursionlimit(10**7)
        
        n = len(nums)
        s = sum(nums)
        
        # Quick edge if s == target => whole array is the answer
        if s == target:
            return n
        
        # Helper: find minimal-length subarray in at most 2 copies of nums that sums to 'x'.
        # Return -1 if none found.  (All positive => standard sliding window for "sum == x".)
        def minSubarraySum(x: int) -> int:
            # We'll use two-pointer across up to 2 copies of nums, total length 2*n.
            # If at any time current_sum > x, we shrink from left.
            # If current_sum == x, update best answer.
            arr_len = 2 * n
            left = 0
            current_sum = 0
            best = float('inf')
            
            for right in range(arr_len):
                current_sum += nums[right % n]
                
                while current_sum > x and left <= right:
                    current_sum -= nums[left % n]
                    left += 1
                
                if current_sum == x:
                    best = min(best, right - left + 1)
            
            return best if best != float('inf') else -1
        
        # Helper: find minimal prefix length (in up to 2 copies) whose sum == x.
        # Return -1 if none. Since it's a prefix, we just add from left to right.
        def minPrefixSum(x: int) -> int:
            total = 0
            for i in range(2*n):  # enough for x up to 2 sums potentially
                total += nums[i % n]
                if total == x:
                    return i + 1   # length is index+1
                if total > x:
                    break
            return -1
        
        # Helper: find minimal suffix length (in up to 2 copies) whose sum == x.
        # Return -1 if none. We'll scan from right to left, or we can do left->right but store prefix sums.
        def minSuffixSum(x: int) -> int:
            total = 0
            # We'll read from the end of "2 copies" backwards.
            # Index from 2*n-1 down to 0
            # But physically we'll do i in reversed(range(2*n)).
            i = 2*n - 1
            length = 0
            while i >= 0:
                total += nums[i % n]
                length += 1
                if total == x:
                    return length
                if total > x:
                    break
                i -= 1
            return -1
        
        # case 2: if s > target => just find in up to 2 copies
        if s > target:
            ans = minSubarraySum(target)
            return ans
        
        # case 3: s < target
        k, r = divmod(target, s)
        
        # 3a) if r == 0 => exactly k copies
        if r == 0:
            return k * n  # no smaller subarray can sum to k*s
        
        # 3b) target = k*s + r
        # candidate1 = k*n + minSubarraySum(r)
        c1_len = minSubarraySum(r)
        candidate1 = (k*n + c1_len) if c1_len != -1 else -1
        
        # candidate2 => remove prefix or suffix of sum = (k+1)*s - target = s-r from (k+1) copies
        # we want to find minimal length prefix or suffix that sums to (s-r).
        sr = s - r
        pref_len = minPrefixSum(sr)
        suff_len = minSuffixSum(sr)
        
        candidate2 = float('inf')
        # if we find a prefix of length L with sum=sr, then the resulting subarray is (k+1)*n - L
        if pref_len != -1:
            candidate2 = min(candidate2, (k+1)*n - pref_len)
        # if we find a suffix of length L with sum=sr, then the resulting subarray is (k+1)*n - L
        if suff_len != -1:
            candidate2 = min(candidate2, (k+1)*n - suff_len)
        
        if candidate2 == float('inf'):
            candidate2 = -1
        
        # final answer is the minimum positive among candidate1, candidate2
        if candidate1 == -1 and candidate2 == -1:
            return -1
        elif candidate1 == -1:
            return candidate2
        elif candidate2 == -1:
            return candidate1
        else:
            return min(candidate1, candidate2)