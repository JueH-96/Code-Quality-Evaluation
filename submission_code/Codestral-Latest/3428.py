class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        from collections import Counter

        # Count the frequency of each number in the array
        count = Counter(nums)

        # Initialize the result to 0
        result = 0

        # Iterate through the count dictionary
        for num, freq in count.items():
            # If the number appears twice, XOR it with the result
            if freq == 2:
                result ^= num

        return result