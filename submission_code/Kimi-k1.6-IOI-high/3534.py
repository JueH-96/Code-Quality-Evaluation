class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def can_swap(num, target):
            s = list(str(num))
            n = len(s)
            if n == 1:
                return False
            for i in range(n):
                for j in range(i + 1, n):
                    new_s = s.copy()
                    new_s[i], new_s[j] = new_s[j], new_s[i]
                    new_num = int(''.join(new_s))
                    if new_num == target:
                        return True
            return False
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                x, y = nums[i], nums[j]
                if x == y:
                    count += 1
                else:
                    if can_swap(x, y) or can_swap(y, x):
                        count += 1
        return count