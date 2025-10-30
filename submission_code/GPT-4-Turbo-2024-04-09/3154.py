class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_value = 0
        found_positive = False
        
        # Iterate over all possible (i, j, k) triplets where i < j < k
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    triplet_value = (nums[i] - nums[j]) * nums[k]
                    if triplet_value > max_value:
                        max_value = triplet_value
                        found_positive = True
        
        # If all values are negative, return 0
        return max_value if found_positive else 0