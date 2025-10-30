class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(n):
            return bin(n).count('1')

        bit_counts = [count_set_bits(num) for num in nums]
        sorted_bit_counts = sorted(bit_counts)

        for i in range(len(bit_counts)):
            if bit_counts[i] != sorted_bit_counts[i]:
                # Check if we can swap adjacent elements with the same number of set bits
                # to sort the bit_counts array
                j = i + 1
                while j < len(bit_counts) and bit_counts[j] == bit_counts[j - 1]:
                    j += 1

                # If we can't find enough elements to swap, return False
                if j - i < abs(i - bit_counts.index(sorted_bit_counts[i])):
                    return False

                # Perform the swaps
                bit_counts[i:j] = sorted(bit_counts[i:j])

        return True