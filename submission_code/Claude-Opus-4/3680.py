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
        
        # Create a map from value to index
        val_to_idx = {num: i for i, num in enumerate(nums)}
        
        # For each possible divisor up to threshold
        for divisor in range(1, threshold + 1):
            # Find all numbers divisible by this divisor
            indices = []
            for multiple in range(divisor, threshold + 1, divisor):
                if multiple in val_to_idx:
                    indices.append(val_to_idx[multiple])
            
            # Connect all pairs if their LCM <= threshold
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    # Calculate LCM
                    a, b = nums[idx1], nums[idx2]
                    gcd_val = self.gcd(a, b)
                    lcm_val = (a * b) // gcd_val
                    if lcm_val <= threshold:
                        union(idx1, idx2)
        
        # Count connected components
        return len(set(find(i) for i in range(n)))
    
    def gcd(self, a, b):
        while b:
            a, b = b, a % b
        return a