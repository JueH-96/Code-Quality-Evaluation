class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def are_almost_equal(x, y):
            sx = str(x)
            sy = str(y)
            if len(sx) != len(sy):
                if x < y:
                    sx = '0' * (len(sy) - len(sx)) + sx
                else:
                    sy = '0' * (len(sx) - len(sy)) + sy
            
            if sx == sy:
                return True
            
            
            def check_swap(s1, s2):
                diff_indices = []
                for i in range(len(s1)):
                    if s1[i] != s2[i]:
                        diff_indices.append(i)
                
                if len(diff_indices) == 0:
                    return True
                elif len(diff_indices) == 2:
                    i, j = diff_indices
                    temp_s1 = list(s1)
                    temp_s1[i], temp_s1[j] = temp_s1[j], temp_s1[i]
                    return "".join(temp_s1) == s2
                else:
                    return False
            
            return check_swap(sx, sy) or check_swap(sy, sx)

        count = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if are_almost_equal(nums[i], nums[j]):
                    count += 1
        return count