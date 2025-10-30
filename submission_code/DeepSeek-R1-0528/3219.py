class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        arr = sorted([(nums[i], i) for i in range(n)])
        
        res = [0] * n
        comp_start = 0
        
        for i in range(1, n + 1):
            if i == n or arr[i][0] - arr[i-1][0] > limit:
                comp = arr[comp_start:i]
                indices = sorted(idx for val, idx in comp)
                values = sorted(val for val, idx in comp)
                for idx_val, idx in enumerate(indices):
                    res[idx] = values[idx_val]
                comp_start = i
        
        return res