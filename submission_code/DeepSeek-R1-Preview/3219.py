class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        if n == 0:
            return []
        arr = sorted((nums[i], i) for i in range(n))
        groups = []
        current_group = [arr[0]]
        
        for i in range(1, n):
            if arr[i][0] - current_group[-1][0] <= limit:
                current_group.append(arr[i])
            else:
                groups.append(current_group)
                current_group = [arr[i]]
        groups.append(current_group)
        
        res = [0] * n
        for group in groups:
            indices = sorted(idx for (val, idx) in group)
            for i in range(len(indices)):
                res[indices[i]] = group[i][0]
        
        return res