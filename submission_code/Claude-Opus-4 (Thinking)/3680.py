class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Check all pairs
        for i in range(n):
            if nums[i] > threshold:
                continue
            for j in range(i + 1, n):
                if nums[j] > threshold:
                    continue
                
                # Calculate LCM using the formula: lcm(a,b) = a*b/gcd(a,b)
                import math
                gcd_val = math.gcd(nums[i], nums[j])
                lcm_val = nums[i] * nums[j] // gcd_val
                
                if lcm_val <= threshold:
                    union(i, j)
        
        # Count unique parents (connected components)
        return len(set(find(i) for i in range(n)))