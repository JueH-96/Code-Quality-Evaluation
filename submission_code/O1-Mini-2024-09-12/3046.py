class Solution:
    def minimumOperations(self, num: str) -> int:
        endings = ['00', '25', '50', '75']
        min_ops = len(num)  # Initialize with the maximum possible deletions

        for end in endings:
            first_char, second_char = end[1], end[0]
            pos1 = -1
            # Find the position of the first character from the end
            for i in range(len(num)-1, -1, -1):
                if num[i] == first_char:
                    pos1 = i
                    break
            if pos1 == -1:
                continue  # This ending is not possible
            # Find the position of the second character before pos1
            pos2 = -1
            for i in range(pos1-1, -1, -1):
                if num[i] == second_char:
                    pos2 = i
                    break
            if pos2 == -1:
                continue  # This ending is not possible
            # Calculate deletions: digits after pos1 and between pos2 and pos1
            deletions = (len(num) - pos1 -1) + (pos1 - pos2 -1)
            min_ops = min(min_ops, deletions)
        
        # Additionally, check if we can make the number '0' by deleting all digits except one '0'
        if '0' in num:
            pos = num.rfind('0')
            deletions = len(num) - 1
            min_ops = min(min_ops, deletions)
        
        return min_ops