class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count_even = 0
        count_odd = 0
        for num in nums:
            if num % 2 == 0:
                count_even += 1
            else:
                count_odd += 1

        # Option 1: All consecutive pairs have the same parity
        max_p0 = max(count_even, count_odd)

        # Option 2: All consecutive pairs have different parity
        def max_alternating(start_parity):
            count = 0
            expected = start_parity
            for num in nums:
                if num % 2 == expected:
                    count += 1
                    expected = 1 - expected
            return count

        max_p1_start_odd = max_alternating(1)
        max_p1_start_even = max_alternating(0)
        max_p1 = max(max_p1_start_odd, max_p1_start_even)

        return max(max_p0, max_p1)