class Solution:
    def countPairs(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        def can_become_equal(x, y):
            str_x = str(x)
            str_y = str(y)
            if len(str_x) != len(str_y):
                return False
            
            count_x = defaultdict(int)
            count_y = defaultdict(int)
            
            for char in str_x:
                count_x[char] += 1
            for char in str_y:
                count_y[char] += 1
            
            # Check if we can match the counts by swapping
            for char in count_x:
                if count_x[char] > count_y[char] + 1 or count_y[char] > count_x[char] + 1:
                    return False
            
            return True
        
        count = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                if can_become_equal(nums[i], nums[j]):
                    count += 1
        
        return count