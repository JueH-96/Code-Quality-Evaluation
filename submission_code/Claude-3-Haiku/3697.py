from collections import defaultdict

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        # Create a dictionary to store the count of each number in nums
        num_count = defaultdict(int)
        for num in nums:
            num_count[num] += 1
        
        # Initialize the total number of operations required
        total_ops = 0
        
        # Iterate through the target list
        for t in target:
            # If the target number is already present in nums, continue to the next target
            if t in num_count and num_count[t] > 0:
                num_count[t] -= 1
                continue
            
            # Find the smallest number in nums that is less than the target
            closest_num = max(num for num in num_count if num < t)
            
            # Increment the closest number until it becomes a multiple of the target
            while closest_num % t != 0:
                closest_num += 1
                total_ops += 1
                
            # Decrement the count of the closest number
            num_count[closest_num] -= 1
            
            # If the closest number is now a multiple of the target, add it to nums
            if closest_num % t == 0:
                num_count[closest_num] += 1
        
        return total_ops