class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        for i in range(n):
            seen = [False] * (n + 2)
            seen[nums[i]] = True
            cur = 0
            for j in range(i + 1, n):
                x = nums[j]
                if not seen[x]:
                    if seen[x - 1] and seen[x + 1]:
                        cur -= 1
                    elif not seen[x - 1] and not seen[x + 1]:
                        cur += 1
                    seen[x] = True
                res += cur
        return res