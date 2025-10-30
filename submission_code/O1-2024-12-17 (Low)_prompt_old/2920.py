class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        n = len(nums)
        if n == 1:
            return 0
        
        # Check if already all equal
        all_equal = True
        for i in range(1, n):
            if nums[i] != nums[0]:
                all_equal = False
                break
        if all_equal:
            return 0
        
        # Map each distinct value to the list of indices where it appears
        positions = defaultdict(list)
        for i, val in enumerate(nums):
            positions[val].append(i)
        
        # For a given list of sorted indices of a value x, we compute
        # the maximum gap between consecutive occurrences in a circular array.
        # The time needed to fill that gap is (gap + 1) // 2.
        # The cost for value x is the max of these times among all gaps.
        # We then take the minimum cost over all distinct values.
        
        answer = float('inf')
        
        for val, inds in positions.items():
            inds.sort()
            
            # Track the maximum gap in a circular manner
            max_gap = 0
            for i in range(len(inds) - 1):
                gap = inds[i+1] - inds[i] - 1
                max_gap = max(max_gap, gap)
            
            # Don't forget the wrap-around gap
            # We add n to the first index to handle circular distance
            wrap_gap = (inds[0] + n) - inds[-1] - 1
            max_gap = max(max_gap, wrap_gap)
            
            time_needed = (max_gap + 1) // 2
            answer = min(answer, time_needed)
        
        return answer