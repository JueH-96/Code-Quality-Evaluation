class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        min_ops = n

        # Target "00"
        first_zero_index = -1
        for i in range(n - 1, -1, -1):
            if num[i] == '0':
                first_zero_index = i
                break
        if first_zero_index != -1:
            for i in range(first_zero_index - 1, -1, -1):
                if num[i] == '0':
                    min_ops = min(min_ops, n - 2)
                    break

        # Target "25"
        last_five_index = -1
        for i in range(n - 1, -1, -1):
            if num[i] == '5':
                last_five_index = i
                break
        if last_five_index != -1:
            for i in range(last_five_index - 1, -1, -1):
                if num[i] == '2':
                    min_ops = min(min_ops, n - 2)
                    break

        # Target "50"
        last_zero_index = -1
        for i in range(n - 1, -1, -1):
            if num[i] == '0':
                last_zero_index = i
                break
        if last_zero_index != -1:
            for i in range(last_zero_index - 1, -1, -1):
                if num[i] == '5':
                    min_ops = min(min_ops, n - 2)
                    break

        # Target "75"
        last_five_index = -1
        for i in range(n - 1, -1, -1):
            if num[i] == '5':
                last_five_index = i
                break
        if last_five_index != -1:
            for i in range(last_five_index - 1, -1, -1):
                if num[i] == '7':
                    min_ops = min(min_ops, n - 2)
                    break

        # Target "0"
        if '0' in num:
            if n > 1 or num[0] != '0':
                min_ops = min(min_ops, n - 1)
            elif n == 1 and num[0] == '0':
                min_ops = min(min_ops, 0)

        return min_ops