class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def are_almost_equal(x, y):
            x_str, y_str = str(x), str(y)
            if len(x_str) != len(y_str):
                return False
            
            diff = sum(a != b for a, b in zip(x_str, y_str))
            return diff == 0 or (diff == 2 and sorted(x_str) == sorted(y_str))

        count = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                if are_almost_equal(nums[i], nums[j]):
                    count += 1
        
        return count