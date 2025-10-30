from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Count the frequency of the most frequent element.
        # Since the array is sorted, we can count by traversing once.
        max_freq = 0
        current_count = 0
        current_val = None
        
        for num in nums:
            if num == current_val:
                current_count += 1
            else:
                max_freq = max(max_freq, current_count)
                current_val = num
                current_count = 1
        max_freq = max(max_freq, current_count)
        
        # The maximum number of removal pairs is limited by:
        # 1. n//2 (because each removal takes 2 elements)
        # 2. The number of elements not from the most frequent group (n - max_freq)
        # If the most frequent element appears more than half the time,
        # then the maximal pairing is (n - max_freq). In that case, the length remaining is:
        # n - 2*(n - max_freq) = 2*max_freq - n.
        # Otherwise, we can remove almost everything, only possibly leaving 1 unpaired element if n is odd.
        if max_freq > n - max_freq:
            return 2 * max_freq - n
        else:
            return n % 2

# Sample test cases:
if __name__ == '__main__':
    sol = Solution()
    print(sol.minLengthAfterRemovals([1, 3, 4, 9]))  # Expected output: 0
    print(sol.minLengthAfterRemovals([2, 3, 6, 9]))  # Expected output: 0
    print(sol.minLengthAfterRemovals([1, 1, 2]))     # Expected output: 1