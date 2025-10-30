class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        min_ops = float('inf')
        # We try to find the optimal number of duplications
        # The number of duplications can be up to the point where the sum exceeds k
        # Since each duplication doubles the count of elements, we can limit the search
        # to the number of times we can duplicate before the sum exceeds k
        # For each possible number of duplications, we calculate the required increments
        # and find the minimum total operations
        # The number of duplications can be up to log2(k) + 1
        max_duplications = 0
        current = 1
        while current < k:
            current *= 2
            max_duplications += 1
        for d in range(max_duplications + 1):
            # After d duplications, the number of elements is 2^d
            num_elements = 2 ** d
            # The sum is num_elements * x, where x is the value of each element
            # We need num_elements * x >= k
            # So x >= ceil(k / num_elements)
            required_x = (k + num_elements - 1) // num_elements
            # The initial value is 1, so the number of increments is required_x - 1
            increments = required_x - 1
            # Total operations is increments + d
            total_ops = increments + d
            if total_ops < min_ops:
                min_ops = total_ops
        return min_ops