class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # Count the number of elements that are 1, 2, and 0 mod 3
        count_mod_1 = sum(1 for num in nums if num % 3 == 1)
        count_mod_2 = sum(1 for num in nums if num % 3 == 2)
        count_mod_0 = sum(1 for num in nums if num % 3 == 0)

        # Calculate the minimum operations needed
        # We can pair up count_mod_1 and count_mod_2 to make them divisible by 3
        # Any remaining count_mod_1 or count_mod_2 will need to be adjusted by 2 operations each
        operations = abs(count_mod_1 - count_mod_2) + (count_mod_1 + count_mod_2) % 2

        return operations