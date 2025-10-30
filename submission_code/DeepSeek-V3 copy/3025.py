class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        from collections import defaultdict

        # Count the frequency of each power of 2 in nums
        num_counts = defaultdict(int)
        for num in nums:
            num_counts[num] += 1

        # Decompose the target into its binary representation
        target_bits = []
        while target > 0:
            target_bits.append(target % 2)
            target = target // 2

        operations = 0

        for i in range(len(target_bits)):
            required = target_bits[i]
            available = num_counts.get(1 << i, 0)

            if available >= required:
                num_counts[1 << i] -= required
            else:
                # Need to split higher powers
                needed = required - available
                # Try to get the needed bits from higher powers
                for j in range(i + 1, 32):
                    if num_counts.get(1 << j, 0) > 0:
                        # Split 1 << j into 2 * (1 << (j-1))
                        num_counts[1 << j] -= 1
                        num_counts[1 << (j-1)] += 2
                        operations += 1
                        # Re-check the current bit
                        available = num_counts.get(1 << i, 0)
                        if available >= needed:
                            num_counts[1 << i] -= needed
                            break
                        else:
                            needed -= available
                            num_counts[1 << i] = 0
                else:
                    # If no higher power can be split, it's impossible
                    return -1

        return operations