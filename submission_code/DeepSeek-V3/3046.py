class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        min_ops = float('inf')
        
        # Iterate over all possible pairs of digits that can form a number divisible by 25
        # The last two digits must be one of the following: 00, 25, 50, 75
        targets = ['00', '25', '50', '75']
        
        for target in targets:
            # Find the positions of the last two digits that match the target
            # We need to find the last occurrence of target[1] and then the last occurrence of target[0] before that
            last_char = target[1]
            second_last_char = target[0]
            
            # Find the last index of last_char
            last_index = -1
            for i in range(n-1, -1, -1):
                if num[i] == last_char:
                    last_index = i
                    break
            
            if last_index == -1:
                continue
            
            # Find the last index of second_last_char before last_index
            second_last_index = -1
            for i in range(last_index-1, -1, -1):
                if num[i] == second_last_char:
                    second_last_index = i
                    break
            
            if second_last_index == -1:
                continue
            
            # Calculate the number of deletions
            # All digits after last_index are deleted
            # All digits between second_last_index and last_index are deleted
            # All digits before second_last_index are kept
            deletions = (n - 1 - last_index) + (last_index - 1 - second_last_index)
            min_ops = min(min_ops, deletions)
        
        # Also consider the case where the number is made to be 0 by deleting all digits
        # The number of deletions is n
        min_ops = min(min_ops, n)
        
        # Also consider the case where the number is already divisible by 25
        # No deletions are needed
        if int(num) % 25 == 0:
            min_ops = 0
        
        return min_ops