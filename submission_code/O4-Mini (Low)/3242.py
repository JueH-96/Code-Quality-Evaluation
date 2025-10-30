from typing import List
from collections import Counter

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count the frequency of each element
        freq = Counter(nums)
        
        # Find the maximum frequency
        max_freq = max(freq.values())
        
        # Sum the counts of elements whose frequency equals the maximum
        total = sum(count for count in freq.values() if count == max_freq)
        
        return total

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxFrequencyElements([1,2,2,3,1,4]))  # Output: 4
    print(sol.maxFrequencyElements([1,2,3,4,5]))    # Output: 5