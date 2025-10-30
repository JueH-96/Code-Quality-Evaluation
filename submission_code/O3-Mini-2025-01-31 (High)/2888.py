from typing import List

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        # Find the dominant (majority) element using Boyer-Moore Voting Algorithm.
        candidate = None
        count = 0
        for num in nums:
            if count == 0:
                candidate = num
                count = 1
            elif num == candidate:
                count += 1
            else:
                count -= 1
        
        # Count the total occurrences of the candidate in the full array.
        total_count = 0
        for num in nums:
            if num == candidate:
                total_count += 1
        
        left_count = 0
        # Try every possible split index i, where 0 <= i < n - 1.
        for i in range(n - 1):
            if nums[i] == candidate:
                left_count += 1
            # Left subarray: nums[0...i] with length (i+1)
            # Right subarray: nums[i+1...n-1] with length (n-i-1)
            # A subarray's candidate is dominant if its frequency * 2 > length.
            if left_count * 2 > (i + 1) and (total_count - left_count) * 2 > (n - i - 1):
                return i
        
        return -1

# The following is provided for simple I/O testing.
def solve():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Convert input tokens to integers.
    nums = list(map(int, data))
    sol = Solution()
    result = sol.minimumIndex(nums)
    sys.stdout.write(str(result))

if __name__ == '__main__':
    solve()