class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        indexed_nums = sorted([(num, i) for i, num in enumerate(nums)])
        
        groups = []
        current_group = [indexed_nums[0]]
        
        for i in range(1, n):
            if indexed_nums[i][0] - indexed_nums[i-1][0] <= limit:
                current_group.append(indexed_nums[i])
            else:
                groups.append(current_group)
                current_group = [indexed_nums[i]]
        groups.append(current_group)
        
        result = [0] * n
        for group in groups:
            values = sorted(num for num, _ in group)
            indices = sorted(idx for _, idx in group)
            for value, index in zip(values, indices):
                result[index] = value
        
        return result