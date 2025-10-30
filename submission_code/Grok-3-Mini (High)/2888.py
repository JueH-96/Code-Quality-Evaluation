class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find the dominant element using Boyer-Moore majority vote
        candidate = nums[0]
        vote_count = 1
        for i in range(1, n):
            if vote_count == 0:
                candidate = nums[i]
                vote_count = 1
            elif nums[i] == candidate:
                vote_count += 1
            else:
                vote_count -= 1
        
        dominant_element = candidate
        
        # Count the total frequency of the dominant element
        total_frequency = 0
        for num in nums:
            if num == dominant_element:
                total_frequency += 1
        
        # Find the smallest index i for a valid split
        left_frequency = 0
        for i in range(n):
            if nums[i] == dominant_element:
                left_frequency += 1
            if i < n - 1:
                length_left = i + 1
                if left_frequency * 2 > length_left:
                    right_frequency = total_frequency - left_frequency
                    length_right = n - i - 1
                    if right_frequency * 2 > length_right:
                        return i
        
        return -1