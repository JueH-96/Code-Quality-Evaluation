class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        # Helper function to check if all digits are less than or equal to limit
        def all_digits_within_limit(number: int) -> bool:
            return all(int(digit) <= limit for digit in str(number))
        
        # The length of the suffix
        suffix_length = len(s)
        # The smallest number with the suffix s
        min_with_suffix = int(s)
        # The smallest number that can be formed with the suffix s and the given limit
        smallest_powerful = min_with_suffix
        while len(str(smallest_powerful)) < suffix_length or not all_digits_within_limit(smallest_powerful):
            smallest_powerful += 10 ** suffix_length
        
        # If the smallest powerful number is greater than finish, there are no powerful numbers in the range
        if smallest_powerful > finish:
            return 0
        
        # The largest number that can be formed with the suffix s and the given limit
        largest_powerful = int(str(limit) * (len(str(finish)) - suffix_length) + s)
        
        # If the largest powerful number is smaller than start, there are no powerful numbers in the range
        if largest_powerful < start:
            return 0
        
        # The number of powerful integers is the difference between the counts of powerful integers up to finish and start-1
        count_finish = (min(finish, largest_powerful) - smallest_powerful) // (10 ** suffix_length) + 1
        count_start = (max(start - 1, smallest_powerful) - smallest_powerful) // (10 ** suffix_length) + 1
        
        return count_finish - count_start + (1 if start <= smallest_powerful <= finish else 0)

# Example usage:
sol = Solution()
print(sol.numberOfPowerfulInt(1, 6000, 4, "124"))  # Output: 5
print(sol.numberOfPowerfulInt(15, 215, 6, "10"))  # Output: 2
print(sol.numberOfPowerfulInt(1000, 2000, 4, "3000"))  # Output: 0