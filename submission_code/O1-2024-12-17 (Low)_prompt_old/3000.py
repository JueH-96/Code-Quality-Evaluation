class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        """
        We want two indices i and j with |i - j| >= x, minimizing |nums[i] - nums[j]|.
        
        Approach:
        
        1) If x == 0, then any distinct pair (i, j) is valid. A well-known method to find
           the minimal absolute difference among all pairs is to sort the array and take
           the minimum difference of consecutive elements in sorted order. This avoids
           comparing the same element (i = j).
        
        2) If x > 0, we can iterate through the array in index order while maintaining
           a balanced "search set" of values from indices up to (i - x). That is, when
           we reach index i, all indices j <= i - x are allowable partners, because
           i - j >= x. We then search in that set for the element(s) closest to nums[i]
           and update the answer. We insert the element nums[i - x] (when i >= x) as
           we move forward.
           
           In Python, we don't have a built-in balanced tree, but we can simulate
           it with a sorted list and bisect for correctness (though, in worst-case,
           insertions can be O(n)). This suffices as a correct solution.
        """
        import bisect

        n = len(nums)
        # Special case: x == 0 => all pairs i != j are valid
        if x == 0:
            sorted_nums = sorted(nums)
            answer = float('inf')
            for i in range(n - 1):
                diff = sorted_nums[i+1] - sorted_nums[i]
                if diff < answer:
                    answer = diff
            return answer
        
        # For x > 0, maintain a sorted structure of elements
        # whose indices are at least x behind the current index i.
        s = []
        answer = float('inf')
        
        for i in range(n):
            # Add the element from index (i - x) if valid
            if i - x >= 0:
                val_to_add = nums[i - x]
                pos = bisect.bisect_left(s, val_to_add)
                s.insert(pos, val_to_add)
            
            # Now find the closest element(s) in s to nums[i]
            if s:
                val = nums[i]
                pos = bisect.bisect_left(s, val)
                
                if pos < len(s):
                    answer = min(answer, abs(s[pos] - val))
                if pos > 0:
                    answer = min(answer, abs(s[pos-1] - val))
        
        return answer