from typing import List
from collections import defaultdict

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        # Compute next greater element to the right (strictly greater).
        nxt = [n] * n
        stack = []
        for i, v in enumerate(nums):
            # Pop while current is strictly greater than stack top
            while stack and nums[stack[-1]] < v:
                idx = stack.pop()
                nxt[idx] = i
            stack.append(i)
        
        # Group indices by value
        positions = defaultdict(list)
        for i, v in enumerate(nums):
            positions[v].append(i)
        
        ans = 0
        # For each value, count valid subarrays among its occurrences
        for v, pos in positions.items():
            # pos is already in increasing order
            m = len(pos)
            # two pointers: for each left index pos[i], find the max right j
            r = 0
            for i in range(m):
                # ensure r >= i
                if r < i:
                    r = i
                # advance r while pos[r] < nextGreater[pos[i]]
                # i.e., no larger element in (pos[i], pos[r]]
                while r + 1 < m and pos[r+1] < nxt[pos[i]]:
                    r += 1
                # all j in [i..r] form valid subarrays with left at pos[i]
                ans += (r - i + 1)
        return ans

# Example usage:
if __name__ == "__main__":
    s = Solution()
    print(s.numberOfSubarrays([1,4,3,3,2]))  # 6
    print(s.numberOfSubarrays([3,3,3]))      # 6
    print(s.numberOfSubarrays([1]))          # 1