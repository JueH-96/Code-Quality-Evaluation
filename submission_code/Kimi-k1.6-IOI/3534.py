class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def can_swap(a: int, b: int) -> bool:
            s = str(a)
            t = str(b)
            if len(s) != len(t):
                return False
            n = len(s)
            for i in range(n):
                for j in range(i + 1, n):
                    lst = list(s)
                    lst[i], lst[j] = lst[j], lst[i]
                    if int(''.join(lst)) == b:
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