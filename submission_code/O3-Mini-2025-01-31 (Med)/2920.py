from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: list) -> int:
        n = len(nums)
        # If all elements are already equal, no seconds are needed.
        if all(x == nums[0] for x in nums):
            return 0
            
        # For each unique value, record all the indices where it appears.
        positions = defaultdict(list)
        for i, num in enumerate(nums):
            positions[num].append(i)
            
        # For each candidate value, we determine the maximum gap (distance)
        # between consecutive occurrences on the circular array.
        # After each second, the candidate "spreads" left and right one index.
        # Thus, the required seconds for that candidate equals ceil(max_gap / 2).
        ans = float('inf')
        for val, idx_list in positions.items():
            # The indices are naturally sorted as we traversed the array in order.
            k = len(idx_list)
            max_gap = 0
            
            # Compute the gap between consecutive occurrences (excluding themselves).
            for i in range(k - 1):
                gap = idx_list[i+1] - idx_list[i] - 1
                if gap > max_gap:
                    max_gap = gap
            
            # Compute the wrap-around gap between the last and first occurrence.
            wrap_gap = (idx_list[0] + n - idx_list[-1] - 1)
            if wrap_gap > max_gap:
                max_gap = wrap_gap
            
            # The seconds needed for candidate value "val" is the ceiling of half the gap.
            seconds_needed = (max_gap + 1) // 2
            ans = min(ans, seconds_needed)
            
        return ans

# Example usage and testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.minimumSeconds([1,2,1,2]))   # Expected output: 1
    # Example 2:
    print(sol.minimumSeconds([2,1,3,3,2])) # Expected output: 2
    # Example 3:
    print(sol.minimumSeconds([5,5,5,5]))   # Expected output: 0