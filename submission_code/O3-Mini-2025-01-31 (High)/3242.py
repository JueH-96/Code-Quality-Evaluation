from typing import List
from collections import Counter

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count the frequency of each element in nums
        freq = Counter(nums)
        # Determine the highest frequency
        max_freq = max(freq.values())
        # Sum the frequencies of the elements that have the maximum frequency
        total = sum(count for count in freq.values() if count == max_freq)
        return total

# Sample test cases to validate the solution
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.maxFrequencyElements([1,2,2,3,1,4]))  # Expected output: 4
    # Example 2:
    print(sol.maxFrequencyElements([1,2,3,4,5]))      # Expected output: 5