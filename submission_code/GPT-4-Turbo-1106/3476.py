class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # Count the remainders when dividing by 3
        remainder_counts = [0, 0, 0]
        for num in nums:
            remainder_counts[num % 3] += 1
        
        # Calculate the number of operations needed
        operations = 0
        while max(remainder_counts) != len(nums):
            for i in range(3):
                if remainder_counts[i] > len(nums) // 3:
                    excess = remainder_counts[i] - len(nums) // 3
                    remainder_counts[i] -= excess
                    remainder_counts[(i + 1) % 3] += excess
                    operations += excess
        return operations