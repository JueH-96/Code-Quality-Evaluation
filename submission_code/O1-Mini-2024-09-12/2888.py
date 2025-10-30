class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        from collections import defaultdict

        n = len(nums)

        # Find the dominant element using Boyer-Moore Voting Algorithm
        candidate = None
        count = 0
        for num in nums:
            if count == 0:
                candidate = num
                count = 1
            elif num == candidate:
                count +=1
            else:
                count -=1

        # Count the total frequency of the candidate
        total_freq = 0
        for num in nums:
            if num == candidate:
                total_freq +=1

        # Iterate to find the minimal split index
        freq_so_far = 0
        for i in range(n-1):
            if nums[i] == candidate:
                freq_so_far +=1
            left_size = i +1
            right_size = n - left_size
            freq_right = total_freq - freq_so_far
            if freq_so_far * 2 > left_size and freq_right * 2 > right_size:
                return i
        return -1