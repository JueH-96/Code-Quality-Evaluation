class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        n = len(nums)
        sorted_nums = sorted(nums)

        def count_set_bits(num):
            return bin(num).count('1')

        current_nums = list(nums)

        for i in range(n):
            target = sorted_nums[i]
            target_set_bits = count_set_bits(target)

            found_index = -1
            for j in range(i, n):
                if current_nums[j] == target and count_set_bits(current_nums[j]) == target_set_bits:
                    found_index = j
                    break

            if found_index == -1:
                return False

            # Move current_nums[found_index] to index i
            for k in range(found_index, i, -1):
                if count_set_bits(current_nums[k]) == count_set_bits(current_nums[k - 1]):
                    current_nums[k], current_nums[k - 1] = current_nums[k - 1], current_nums[k]
                else:
                    return False

        return True