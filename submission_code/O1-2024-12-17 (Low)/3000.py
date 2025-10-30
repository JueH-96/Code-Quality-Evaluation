class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        import bisect

        # We'll maintain a sorted list of elements whose indices
        # are at least x behind the current index i. That is,
        # we want to store nums[j] for all j <= i - x (and j >= 0).
        # Then for each nums[i], we find the closest element in
        # that sorted list (if it exists) and update the answer
        # with the minimum absolute difference.
        
        sorted_list = []
        ans = float('inf')
        
        for i in range(len(nums)):
            # Insert nums[i-x] into the sorted list if i-x >= 0
            # so that sorted_list contains all nums[j] with j <= i-x.
            if i - x >= 0:
                val_to_insert = nums[i - x]
                # Find insertion position.
                pos = bisect.bisect_left(sorted_list, val_to_insert)
                # Insert into sorted_list (O(n) worst-case in plain list).
                sorted_list.insert(pos, val_to_insert)
            
            # Now, find the position of nums[i] in sorted_list
            # and compare with nearest neighbors to update ans.
            if sorted_list:
                pos = bisect.bisect_left(sorted_list, nums[i])
                
                # Check the element at 'pos' if within range
                if pos < len(sorted_list):
                    ans = min(ans, abs(sorted_list[pos] - nums[i]))
                
                # Check the element just before 'pos' if pos > 0
                if pos > 0:
                    ans = min(ans, abs(sorted_list[pos - 1] - nums[i]))
        
        return ans