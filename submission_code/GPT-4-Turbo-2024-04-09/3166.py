class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each number in nums
        frequency = Counter(nums)
        
        # Get the list of frequencies
        counts = list(frequency.values())
        
        # Sort the counts to handle them from smallest to largest
        counts.sort()
        
        # We need at least as many groups as the maximum frequency of any number
        max_groups = max(counts)
        
        # We will use a greedy approach to determine the minimum number of groups
        # that can be formed such that the difference between the number of indices
        # in any two groups does not exceed 1.
        
        # Initialize the number of groups to the maximum frequency
        num_groups = max_groups
        
        # Try to minimize the number of groups needed
        while True:
            # Calculate the number of indices that can be placed in each group
            # if we have num_groups groups
            indices_per_group = len(nums) // num_groups
            remainder = len(nums) % num_groups
            
            # Check if we can distribute the counts into these groups
            # such that no group has more than indices_per_group + 1 indices
            # and the difference between any two groups does not exceed 1
            possible = True
            current_group_capacity = indices_per_group + (1 if remainder > 0 else 0)
            for count in counts:
                if count > current_group_capacity:
                    possible = False
                    break
                remainder -= 1
                if remainder < 0:
                    current_group_capacity = indices_per_group
            
            if possible:
                return num_groups
            else:
                num_groups += 1