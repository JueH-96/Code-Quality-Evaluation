from typing import List

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        aliceIndex = -1
        # Find aliceIndex where nums[aliceIndex] == 1, pick it up initially
        # For simplicity, assume aliceIndex = 0, but we need to find it
        # Actually, aliceIndex can be any index, but the problem allows choosing any aliceIndex
        # To minimize moves, choose aliceIndex that optimizes the distance to ones
        # However, to simplify, we can consider aliceIndex as given or iterate over possible aliceIndex
        # For this solution, assume aliceIndex is given or precomputed
        # For now, assume aliceIndex = 0
        aliceIndex = 0
        initial_one = 0
        if nums[aliceIndex] == 1:
            initial_one = 1
            nums[aliceIndex] = 0  # Picked up, set to 0

        # Calculate distances for ones and zeros
        A = []  # distances for ones
        B = []  # distances for zeros + 1 (flip)
        for i in range(n):
            if i == aliceIndex:
                continue
            if nums[i] == 1:
                dist = abs(i - aliceIndex)
                A.append(dist)
            else:
                dist = abs(i - aliceIndex)
                B.append(1 + dist)  # 1 for flip, dist for swap

        A.sort()
        B.sort()

        # Binary search on the number of moves m
        # We need to find the smallest m such that count_A + min(count_B <= m, maxChanges) >= k - initial_one
        left = 0
        right = max(A + B) if A + B else 0
        target = k - initial_one
        if target <= 0:
            return 0  # No moves needed if initial_one >= k

        def is_possible(m):
            # Count ones that can be collected without flipping
            count_A = 0
            # Binary search to find how many in A are <= m
            # Since A is sorted, use bisect_right
            import bisect
            count_A = bisect.bisect_right(A, m)
            # Count zeros that can be flipped and collected within m moves
            # Need 1 + distance <= m
            # B is already 1 + distance, so count how many B <= m
            count_B = bisect.bisect_right(B, m)
            # Use min(count_B, maxChanges)
            used_B = min(count_B, maxChanges)
            total = count_A + used_B
            return total >= target

        # Perform binary search
        while left < right:
            mid = (left + right) // 2
            if is_possible(mid):
                right = mid
            else:
                left = mid + 1
        return left