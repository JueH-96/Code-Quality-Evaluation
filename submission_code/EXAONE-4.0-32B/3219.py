class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        arr = [(nums[i], i) for i in range(n)]
        arr.sort()
        
        components = []
        cur_comp = []
        for i in range(n):
            if i == 0:
                cur_comp.append(arr[i])
            else:
                if arr[i][0] - arr[i-1][0] <= limit:
                    cur_comp.append(arr[i])
                else:
                    components.append(cur_comp)
                    cur_comp = [arr[i]]
        if cur_comp:
            components.append(cur_comp)
        
        res = [0] * n
        for comp in components:
            indices = sorted(idx for _, idx in comp)
            vals = [val for val, _ in comp]
            for j, idx in enumerate(indices):
                res[idx] = vals[j]
                
        return res