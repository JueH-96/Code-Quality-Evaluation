class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def almost_equal(x, y):
            sx = str(x)
            sy = str(y)
            if sorted(sx) == sorted(sy):
                return True
            
            for i in range(len(sx)):
                for j in range(i + 1, len(sx)):
                    temp = list(sx)
                    temp[i], temp[j] = temp[j], temp[i]
                    if sorted("".join(temp)) == sorted(sy):
                        return True

            for i in range(len(sy)):
                for j in range(i + 1, len(sy)):
                    temp = list(sy)
                    temp[i], temp[j] = temp[j], temp[i]
                    if sorted("".join(temp)) == sorted(sx):
                        return True
            return False

        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if almost_equal(nums[i], nums[j]):
                    count += 1
        return count