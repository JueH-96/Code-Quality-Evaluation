class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def almost_equal(x, y):
            sx, sy = str(x), str(y)
            if len(sx) != len(sy):
                return False
            diff = []
            for i in range(len(sx)):
                if sx[i] != sy[i]:
                    diff.append((sx[i], sy[i]))
                if len(diff) > 2:
                    return False
            return len(diff) == 0 or len(diff) == 2 and diff[0] == diff[1][::-1]
        
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i + 1, n):
                if almost_equal(nums[i], nums[j]):
                    ans += 1
        return ans