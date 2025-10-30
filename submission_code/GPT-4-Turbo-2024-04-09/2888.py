class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        from collections import Counter
        
        n = len(nums)
        if n < 2:
            return -1
        
        # Count frequencies of each number in the entire array
        total_count = Counter(nums)
        
        # Find the dominant element in the entire array
        dominant_element = None
        for num, count in total_count.items():
            if count * 2 > n:
                dominant_element = num
                break
        
        # If no dominant element, although the problem guarantees one, return -1
        if dominant_element is None:
            return -1
        
        # Count frequencies from the left
        left_count = Counter()
        valid_split_index = -1
        
        for i in range(n - 1):
            left_count[nums[i]] += 1
            
            # Check if the current element in the left part can be dominant
            if left_count[nums[i]] * 2 > (i + 1):
                left_dominant = nums[i]
            else:
                continue
            
            # Check if the current element in the right part can be dominant
            right_dominant = None
            right_count = total_count[nums[i]] - left_count[nums[i]]
            if right_count * 2 > (n - i - 1):
                right_dominant = nums[i]
            
            # If both parts have the same dominant element as the whole array's dominant element
            if left_dominant == right_dominant == dominant_element:
                valid_split_index = i
                break
        
        return valid_split_index