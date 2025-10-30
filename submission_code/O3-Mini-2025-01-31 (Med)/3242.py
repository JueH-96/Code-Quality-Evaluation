from typing import List
from collections import Counter

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count the frequency of each element in nums
        count = Counter(nums)
        # Determine the maximum frequency value
        max_freq = max(count.values())
        # Sum total frequencies for elements that have the maximum frequency
        total = sum(freq for freq in count.values() if freq == max_freq)
        return total

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxFrequencyElements([1,2,2,3,1,4]))  # Output: 4
    print(sol.maxFrequencyElements([1,2,3,4,5]))    # Output: 5