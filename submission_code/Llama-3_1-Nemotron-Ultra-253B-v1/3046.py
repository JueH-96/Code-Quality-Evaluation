class Solution:
    def minimumOperations(self, num: str) -> int:
        ans = len(num)  # Initialize with deleting all digits
        targets = ["00", "25", "50", "75"]
        
        for target in targets:
            first, second = target[0], target[1]
            # Find the last occurrence of 'second'
            second_pos = -1
            for i in reversed(range(len(num))):
                if num[i] == second:
                    second_pos = i
                    break
            if second_pos == -1:
                continue
            # Find the last occurrence of 'first' before 'second_pos'
            first_pos = -1
            for i in reversed(range(second_pos)):
                if num[i] == first:
                    first_pos = i
                    break
            if first_pos == -1:
                continue
            # Calculate deletions needed
            current_del = len(num) - first_pos - 2
            ans = min(ans, current_del)
        
        # Check for single '0' case
        if '0' in num:
            ans = min(ans, len(num) - 1)
        
        return ans