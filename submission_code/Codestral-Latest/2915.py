class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        prefix_count = {0: 1}
        current_count = 0
        result = 0

        for num in nums:
            if num % modulo == k:
                current_count = (current_count + 1) % modulo

            if (current_count - k) % modulo in prefix_count:
                result += prefix_count[(current_count - k) % modulo]

            if current_count in prefix_count:
                prefix_count[current_count] += 1
            else:
                prefix_count[current_count] = 1

        return result