from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # Count occurrences of each number
        count_dict = Counter(nums)
        total_operations = 0
        
        for num, count in count_dict.items():
            if count == 1:
                return -1  # Can't delete single occurrences
            
            # Calculate minimum operations based on remainder when divided by 3
            r = count % 3
            if r == 0:
                # If divisible by 3, use only groups of 3
                total_operations += count // 3
            elif r == 1:
                # If remainder 1, use one fewer group of 3 and two groups of 2
                # (count-4)//3 + 2 simplifies to (count+2)//3
                total_operations += (count + 2) // 3
            else:  # r == 2
                # If remainder 2, use one group of 2 and the rest groups of 3
                # (count-2)//3 + 1 simplifies to (count+1)//3
                total_operations += (count + 1) // 3
        
        return total_operations