class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        from math import gcd
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        # Filter numbers <= threshold, others are isolated components
        valid_nums = [x for x in nums if x <= threshold]
        isolated_count = len(nums) - len(valid_nums)
        
        if not valid_nums:
            return isolated_count
        
        # Union-Find for valid numbers
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Initialize all valid numbers
        for num in valid_nums:
            find(num)
        
        # Connect numbers whose LCM <= threshold
        n = len(valid_nums)
        for i in range(n):
            for j in range(i + 1, n):
                if lcm(valid_nums[i], valid_nums[j]) <= threshold:
                    union(valid_nums[i], valid_nums[j])
        
        # Count unique components among valid numbers
        components = len(set(find(num) for num in valid_nums))
        
        return components + isolated_count