from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0
        
        # Build a map: candidate value -> list of indices where it occurs.
        pos_map = {}
        for i, num in enumerate(nums):
            if num not in pos_map:
                pos_map[num] = []
            pos_map[num].append(i)
        
        # If the entire array is already uniform, answer is 0.
        for positions in pos_map.values():
            if len(positions) == n:
                return 0
        
        # For each candidate (which must be the eventual equal value),
        # we can use the allowed operation to “spread” its value along the circle.
        # In one second, each occurrence influences its immediate neighbors.
        # Thus after t seconds, every index within circular distance t of some initial occurrence would become that candidate.
        # The minimum time needed for a candidate x is the maximum (over all indices) of the circular distance
        # from that index to the nearest occurrence of x.
        #
        # It turns out that if you know the positions where x occurs, the worst‐case distance is attained
        # at the midpoint of the largest “gap” between two consecutive occurrences (in the circular sense).
        # For consecutive occurrences at indices p and q, there are (q-p-1) indices between them, and the worst-case
        # distance is ceil((q-p-1)/2). It is equivalent to (q-p)//2.
        # Also, we must consider the wrap‐around gap between the last occurrence and the first occurrence.
        
        ans = float('inf')
        for positions in pos_map.values():
            # positions are collected in increasing order (because we traverse left-to-right)
            max_gap = 0
            # Check the gap between every pair of consecutive occurrences.
            for i in range(len(positions) - 1):
                gap = positions[i+1] - positions[i]
                if gap > max_gap:
                    max_gap = gap
            # Also check the wrap‐around gap.
            wrap_gap = positions[0] + n - positions[-1]
            if wrap_gap > max_gap:
                max_gap = wrap_gap
            # The seconds needed for candidate x is max_gap // 2.
            candidate_time = max_gap // 2
            if candidate_time < ans:
                ans = candidate_time
        
        return ans


# Example usage:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.minimumSeconds([1,2,1,2]))  # Expected output: 1
    # Example 2:
    print(sol.minimumSeconds([2,1,3,3,2]))  # Expected output: 2
    # Example 3:
    print(sol.minimumSeconds([5,5,5,5]))  # Expected output: 0