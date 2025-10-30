class Solution:
    def minimumOperations(self, num: str) -> int:
        num = list(map(int, num))
        operations = 0
        while num:
            min_val = min(num)
            num = [i - min_val for i in num if i > min_val]
            operations += min_val
            if num:
                num = [i for i in num if i > 0]
                operations += 1
        return operations