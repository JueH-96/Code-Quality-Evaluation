class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # First, find the dominant element (which we call 'dom').
        # Since there's exactly one dominant element, we can either
        # use a frequency dictionary or Boyer-Moore voting.
        
        # Approach: Using a frequency dict (sufficient for n <= 10^5)
        from collections import Counter
        
        freq = Counter(nums)
        n = len(nums)
        # The dominant element 'dom' has freq(dom)*2 > n
        dom = max(freq, key=freq.get)
        
        # Build a prefix array p where p[i] = count of 'dom' in nums[:i+1].
        p = [0] * n
        p[0] = 1 if nums[0] == dom else 0
        for i in range(1, n):
            p[i] = p[i-1] + (1 if nums[i] == dom else 0)
        
        total_dom = p[-1]  # total occurrences of dom in the entire array
        
        # We want to find the smallest i in [0..n-2] such that:
        #   Condition1: p[i] * 2 > (i + 1)
        #   Condition2: (total_dom - p[i]) * 2 > (n - (i + 1))
        for i in range(n - 1):
            left_count = p[i]
            right_count = total_dom - left_count
            # Check dominance in left subarray [0..i]
            if left_count * 2 > (i + 1):
                # Check dominance in right subarray [i+1..n-1]
                if right_count * 2 > (n - (i + 1)):
                    return i
        
        return -1