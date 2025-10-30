class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Increase each element by 1
        nums = [num + 1 for num in nums]
        
        # Count the frequency of each element
        count = Counter(nums)
        
        max_consecutive = 0
        current_consecutive = 0
        previous_num = None
        
        # Iterate through the sorted unique elements
        for num in sorted(count.keys()):
            if previous_num is None or num == previous_num + 1:
                current_consecutive += count[num]
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = count[num]
            previous_num = num
        
        max_consecutive = max(max_consecutive, current_consecutive)
        
        return max_consecutive