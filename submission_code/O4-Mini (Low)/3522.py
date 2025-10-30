from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        # Edge case: window size 1 always trivially sorted/consecutive
        if k == 1:
            return nums[:]
        
        # Precompute differences: diff[i] = 1 if nums[i+1] - nums[i] == 1 else 0
        diff = [1 if nums[i+1] - nums[i] == 1 else 0 for i in range(n-1)]
        
        # Build prefix sums over diff to allow O(1) window checks
        # ps[0] = 0, ps[i] = sum(diff[0..i-1])
        ps = [0] * (n)
        for i in range(1, n):
            ps[i] = ps[i-1] + diff[i-1]
        
        res = []
        # For each window start i from 0 to n-k
        # We need exactly k-1 consecutive "1"s in diff[i..i+k-2]
        for i in range(n - k + 1):
            # sum of diff in window = ps[i+k-1] - ps[i]
            if ps[i + k - 1] - ps[i] == k - 1:
                # all transitions are +1, so it's sorted ascending by 1
                res.append(nums[i + k - 1])
            else:
                res.append(-1)
        
        return res

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.resultsArray([1,2,3,4,3,2,5], 3))  # [3,4,-1,-1,-1]
    print(sol.resultsArray([2,2,2,2,2], 4))      # [-1,-1]
    print(sol.resultsArray([3,2,3,2,3,2], 2))    # [-1,3,-1,3,-1]