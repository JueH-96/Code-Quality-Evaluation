class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(n):
            count = 0
            while n > 0:
                n &= (n - 1)
                count += 1
            return count

        groups = []
        current_group = []
        current_bits = -1

        for num in nums:
            bits = count_set_bits(num)
            if bits == current_bits:
                current_group.append(num)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [num]
                current_bits = bits
        
        if current_group:
            groups.append(current_group)

        sorted_nums = []
        for group in groups:
            sorted_nums.extend(sorted(group))

        return sorted_nums == sorted(nums)