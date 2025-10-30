class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def is_perfect_square(n):
            if n < 0:
                return False
            root = int(n**0.5)
            return root * root == n

        def check_complete(subset):
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    if not is_perfect_square(subset[i] * subset[j]):
                        return False
            return True

        n = len(nums)
        max_sum = 0

        for i in range(1 << n):
            subset_indices = []
            subset_elements = []
            current_sum = 0

            for j in range(n):
                if (i >> j) & 1:
                    subset_indices.append(j + 1)
                    subset_elements.append(nums[j])
                    current_sum += nums[j]
            
            if len(subset_elements) > 0 and check_complete(subset_elements):
                max_sum = max(max_sum, current_sum)

        return max_sum