class Solution:
    def minEnd(self, n: int, x: int) -> int:
        nums = [x]
        for _ in range(n - 1):
            last_num = nums[-1]
            current_num = last_num + 1
            while (current_num & x) != x:
                current_num += 1
            nums.append(current_num)
        return nums[-1]