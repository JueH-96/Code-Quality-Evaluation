class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        # Check if the number is already special
        if int(num) % 25 == 0:
            return 0

        # Check for the last two digits
        if int(num[-2:]) % 25 == 0:
            return 0

        # Check for the last three digits
        if int(num[-3:]) % 25 == 0:
            return 1

        # Check for the last four digits
        if int(num[-4:]) % 25 == 0:
            return 2

        # Check for the last five digits
        if int(num[-5:]) % 25 == 0:
            return 3

        # If none of the above, we need to delete more digits
        # We need to find the minimum number of deletions to make the number special
        min_operations = float('inf')

        # Check for the last two digits
        for i in range(n):
            for j in range(i + 1, n):
                new_num = num[:i] + num[i+1:j] + num[j+1:]
                if int(new_num) % 25 == 0:
                    min_operations = min(min_operations, n - len(new_num))

        return min_operations if min_operations != float('inf') else n