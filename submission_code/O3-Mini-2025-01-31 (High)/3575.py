from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp1[i][r] will store the set of possible OR values you can get
        # by choosing exactly r numbers from the prefix nums[0:i]
        dp1 = [[set() for _ in range(k+1)] for _ in range(n+1)]
        dp1[0][0].add(0)
        for i in range(n):
            # first copy over the values when skipping nums[i]
            for r in range(k+1):
                dp1[i+1][r].update(dp1[i][r])
            # then update if we choose nums[i]
            for r in range(k):
                for v in dp1[i][r]:
                    dp1[i+1][r+1].add(v | nums[i])
                    
        # dp2[i][r] will store the set of possible OR values from the suffix
        # nums[i:] by selecting exactly r elements.
        dp2 = [[set() for _ in range(k+1)] for _ in range(n+1)]
        dp2[n][0].add(0)
        for i in range(n-1, -1, -1):
            # if we skip nums[i], just carry the already computed values
            for r in range(k+1):
                dp2[i][r].update(dp2[i+1][r])
            # if we take nums[i]
            for r in range(1, k+1):
                for v in dp2[i+1][r-1]:
                    dp2[i][r].add(v | nums[i])
        
        # Now choose a split point "i" where the first k elements of our 
        # subsequence are chosen from nums[0:i] and the second k are chosen from nums[i:].
        ans = 0
        for i in range(n+1):
            if not dp1[i][k] or not dp2[i][k]:
                continue
            for a in dp1[i][k]:
                for b in dp2[i][k]:
                    ans = max(ans, a ^ b)
        return ans

# Example usage:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.maxValue([2, 6, 7], 1))   # Expected output: 5
    # Example 2:
    print(sol.maxValue([4, 2, 5, 6, 7], 2))  # Expected output: 2