from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        # 1) Compute current maximum adjacent diff between two known (non -1) neighbors
        curMax = 0
        for i in range(n - 1):
            if nums[i] != -1 and nums[i+1] != -1:
                curMax = max(curMax, abs(nums[i] - nums[i+1]))
        
        # 2) Collect all known neighbors that border a missing slot
        A = []
        for i in range(n - 1):
            if nums[i] == -1 and nums[i+1] != -1:
                A.append(nums[i+1])
            elif nums[i] != -1 and nums[i+1] == -1:
                A.append(nums[i])
        
        # If there are no missing borders, answer is just curMax
        if not A:
            return curMax
        
        A.sort()
        
        # 3) We want the 2-center radius R that covers all points in A with at most 2 intervals of length 2R
        #    Binary search R
        def can_cover(R: int) -> bool:
            # greedy cover with at most 2 intervals of length 2R
            intervals_used = 0
            i = 0
            m = len(A)
            while i < m and intervals_used < 2:
                intervals_used += 1
                # interval starts at A[i], covers up to A[i] + 2R
                cover_to = A[i] + 2*R
                # advance i beyond this interval
                while i < m and A[i] <= cover_to:
                    i += 1
            return i >= m
        
        # bounds for R: [0, max(A)-min(A)]
        lo, hi = 0, A[-1] - A[0]
        bestR = hi
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_cover(mid):
                bestR = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        # final answer is max of existing adj-diff and this radius
        return max(curMax, bestR)


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minDifference([1,2,-1,10,8]))  # 4
    print(sol.minDifference([-1,-1,-1]))     # 0
    print(sol.minDifference([-1,10,-1,8]))   # 1