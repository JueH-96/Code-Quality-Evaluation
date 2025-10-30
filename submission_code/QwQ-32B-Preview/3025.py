class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        from math import log2, ceil, floor

        # Create frequency map of nums
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1

        # Get the binary representation of target and identify needed powers
        needed = []
        temp = target
        exponent = 0
        while temp > 0:
            if temp & 1:
                needed.append(1 << exponent)
            temp >>= 1
            exponent += 1

        # Sort needed powers
        needed.sort()

        # Initialize operations counter
        operations = 0

        # Iterate through needed powers
        for need in needed:
            # Check if needed power is available
            if freq.get(need, 0) > 0:
                freq[need] -= 1
            else:
                # Find the smallest power in freq that is >= need
                candidate = None
                for key in sorted(freq.keys(), reverse=True):
                    if key >= need:
                        candidate = key
                        break
                if candidate is None:
                    return -1
                # Split candidate down to need
                current = candidate
                while current > need:
                    # Split current into two halves
                    half = current // 2
                    # Consume one candidate
                    freq[current] -= 1
                    # Add two halves
                    freq[half] = freq.get(half, 0) + 2
                    operations += 1
                    current = half
                # Now, use the needed power
                freq[need] -= 1

        return operations