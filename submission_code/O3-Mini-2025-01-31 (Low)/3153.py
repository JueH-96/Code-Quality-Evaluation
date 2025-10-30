class Solution:
    def maxSum(self, nums: list, k: int) -> int:
        MOD = 10**9 + 7
        # We can note that the allowed operations preserve the total count of ones for each bit position.
        # For each bit position b (0-indexed), let count_b be the number of numbers in nums with the bᵗʰ bit = 1.
        # When performing the allowed operations, you cannot increase the total number of ones for bit b.
        # Moreover, because in any number each bit can be either 0/1, at most count_b of the final numbers can have bit b = 1.
        #
        # We want to select k numbers with maximum possible sum of squares.
        # We are free to reassign the bits among the final numbers provided that for each bit b,
        # the bit 2^b appears in at most count_b of the chosen numbers.
        #
        # Since the square operation is super-additive (due to cross terms when bits are in the same number),
        # it is best to concentrate the bits as much as possible.
        #
        # The optimal strategy is to "pack" the bits into numbers in descending order.
        # Suppose we form k numbers, and for j from 1 to k, we create one number (call it x_j)
        # that gets all bits b for which count_b >= j.
        # Then, clearly, every bit b is used at most count_b times (namely in numbers x_1, x_2, ..., up to x_{count_b})
        # and the sum of squares is:
        #     sum_{j=1}^{k} (value_j)^2, where value_j = sum_{b: count_b >= j} 2^b.
        #
        # This can be shown to be optimal.
        #
        # Count the number of ones for each bit:
        maxBit = 31  # since nums[i] <= 10^9 < 2^31
        bitCount = [0] * maxBit
        for num in nums:
            # iterate through bits 0..30:
            for b in range(maxBit):
                # check if b'th bit is set
                if num & (1 << b):
                    bitCount[b] += 1
        
        # Now for each j from 1 to k, compute the value of the j-th largest "number" we can form.
        # For j-th number, add 2^b for each bit for which bitCount[b] >= j.
        values = []
        for j in range(1, k+1):
            current = 0
            for b in range(maxBit):
                if bitCount[b] >= j:
                    current += (1 << b)
            values.append(current)
        
        result = 0
        for val in values:
            result = (result + (val % MOD) * (val % MOD)) % MOD
        return result