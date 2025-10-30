class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        
        # Create a dictionary to store the indices of each number
        indices = {}
        for i, num in enumerate(nums):
            if num not in indices:
                indices[num] = i
        
        max_length = 0
        for i in range(n):
            num = nums[i]
            if num in indices:
                j = indices[num]
                power = 0
                while j >= 0 and num == nums[j]:
                    if (num ** (2 ** power)) in indices and indices[num ** (2 ** power)] == j:
                        max_length = max(max_length, 2 * power + 1)
                        j -= 1
                        power += 1
                    else:
                        break
        
        return max_length