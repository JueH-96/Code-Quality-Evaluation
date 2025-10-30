class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def get_neighbors(x):
            s = list(str(x))
            n = len(s)
            res = {x}
            for i in range(n):
                for j in range(i+1, n):
                    s[i], s[j] = s[j], s[i]
                    num_str = ''.join(s)
                    num_val = int(num_str)
                    res.add(num_val)
                    s[i], s[j] = s[j], s[i]
            return res
        
        n = len(nums)
        neighbors = [get_neighbors(num) for num in nums]
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if nums[j] in neighbors[i] or nums[i] in neighbors[j]:
                    count += 1
        return count