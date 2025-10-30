from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        m = len(nums)
        total = sum(nums)
        
        # Case 1: target is not greater than total.
        if target <= total:
            best = float("inf")
            curr = 0
            left = 0
            for right in range(m):
                curr += nums[right]
                # Because all numbers are positive, we can slide the window.
                while curr > target and left <= right:
                    curr -= nums[left]
                    left += 1
                if curr == target:
                    best = min(best, right - left + 1)
            return best if best != float("inf") else -1
        
        # For target > total the subarray must span copies.
        # We write target = q * total + r, with 0 <= r < total.
        q, r = divmod(target, total)
        
        # When r == 0, simply taking q full copies (a contiguous block equal to the array repeated)
        # is a valid solution.
        if r == 0:
            return q * m

        # Our analysis shows that any valid contiguous subarray taken from the infinite repetition must be of one of two types.
        # Represent a subarray that starts at some index in one copy and ends at some index in a later copy.
        # Write the choice in terms of the prefix sums of one copy.
        # Let prefix[0] = 0 and for 1 <= i <= m, define prefix[i] = nums[0]+...+nums[i-1].
        # Then any subarray starting in a copy at some position i and ending in a later copy at position j (with i, j in [0, m])
        # has sum = (total - prefix[i]) + prefix[j] plus some multiple of total.
        # In our case, we need some x >= 0 such that:
        #       (total - prefix[i]) + prefix[j] + x * total = target.
        # Write target = q * total + r. Then, rearrange:
        #       (total - prefix[i]) + prefix[j] = target - x*total = (q - x)*total + r.
        # For the left side only two possibilities occur (because its value lies between 0 and 2*total)
        # so that r + (prefix[i] - prefix[j]) equals 0 or equals total.
        #
        # Define two candidate cases:
        # Candidate A (non‐wrap in the same copy “inside” a copy):
        #    If r + (prefix[i] - prefix[j]) = 0 with i <= j,
        #    then prefix[j] - prefix[i] = r.
        #    In that case x = q - 1 and the complete subarray spans (q - 1) full copies plus the subarray from i to j.
        #    Its length is: (m - i) + j + (q - 1) * m = q * m + (j - i).
        # Candidate B (wrap‐around, i.e. the starting index is later in one copy than where the finishing index is in the next copy):
        #    If r + (prefix[i] - prefix[j]) = total with i > j,
        #    then prefix[i] - prefix[j] = total - r.
        #    In that case x = q and the complete subarray has length: (m - i) + j + q * m.
        
        # Precompute prefix sums for a single copy.
        prefix = [0] * (m + 1)
        for i in range(1, m + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]
        
        ans = float("inf")
        
        # Candidate A: look for contiguous segment in one copy with sum = r.
        # Because the numbers are positive we can use a sliding window.
        bestA = float("inf")
        curr = 0
        left = 0
        for right in range(m):
            curr += nums[right]
            while curr > r and left <= right:
                curr -= nums[left]
                left += 1
            if curr == r:
                # In prefix notation, the subarray starting at index "left" and ending at index "right" corresponds
                # to prefix indices i = left and j = right + 1; its length is (right - left + 1)
                bestA = min(bestA, right - left + 1)
        if bestA != float("inf"):
            # For Candidate A, subarray length overall becomes:
            candidateA = q * m + bestA   # (q - 1)*m + (m + (j - i)) = q*m + (j - i)
            ans = min(ans, candidateA)
        
        # Candidate B: look for indices (i, j) with 0 <= j < i <= m such that prefix[i] - prefix[j] = total - r.
        # Then the overall length is (m - i) + j + q * m.
        bestB = float("inf")
        # We can put the prefix values into a dictionary mapping value -> index.
        d = {}
        for j in range(m + 1):
            # Because prefix is strictly increasing (all nums positive) each value appears once.
            d[prefix[j]] = j
        
        for i in range(1, m + 1):
            needed = prefix[i] - (total - r)
            if needed in d:
                j = d[needed]
                if j < i:
                    candidateB = (m - i) + j + q * m
                    bestB = min(bestB, candidateB)
        if bestB != float("inf"):
            ans = min(ans, bestB)
        
        return ans if ans != float("inf") else -1