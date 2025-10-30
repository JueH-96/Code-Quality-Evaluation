class Solution:
    def minimumOperations(self, num: str) -> int:
        min_ops = len(num) + 1  # Initialize with a large number
        targets = ['00', '25', '50', '75']
        for target in targets:
            i = len(num) - 1
            while i >= 0:
                if num[i] == target[1]:
                    j = i - 1
                    while j >= 0:
                        if num[j] == target[0]:
                            deletions = (len(num) - i - 1) + (i - j - 1)
                            min_ops = min(min_ops, deletions)
                            break
                        j -= 1
                i -= 1
        # Check if we can reduce num to '0'
        if '0' in num:
            deletions = len(num) - 1  # Delete all digits except one '0'
            min_ops = min(min_ops, deletions)
        if min_ops == len(num) + 1:
            min_ops = len(num)  # Delete all digits to get '0'
        return min_ops