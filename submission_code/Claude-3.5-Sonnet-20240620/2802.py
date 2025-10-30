class Solution:
    def punishmentNumber(self, n: int) -> int:
        def can_partition(num: int, target: int) -> bool:
            if num == target:
                return True
            if num < target:
                return False
            
            str_num = str(num)
            for i in range(1, len(str_num)):
                left = int(str_num[:i])
                right = int(str_num[i:])
                if left > target:
                    break
                if can_partition(right, target - left):
                    return True
            return False

        result = 0
        for i in range(1, n + 1):
            if can_partition(i * i, i):
                result += i * i
        return result