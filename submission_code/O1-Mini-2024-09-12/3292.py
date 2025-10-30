from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        from collections import defaultdict

        n = len(nums)
        m = len(changeIndices)
        index_positions = defaultdict(list)
        for idx, val in enumerate(changeIndices, 1):
            index_positions[val].append(idx)

        def can_mark(S):
            deadlines = []
            for i in range(1, n+1):
                if i not in index_positions:
                    return False
                # Find first occurrence <= S
                pos = -1
                for t in index_positions[i]:
                    if t <= S:
                        pos = t
                        break
                if pos == -1:
                    return False
                deadlines.append((pos, nums[i-1]))
            # Sort by deadline
            deadlines.sort()
            # Initialize time slots
            slots = [0]*S
            reserved = set()
            for deadline, _ in deadlines:
                reserved.add(deadline-1)
            for deadline, req in deadlines:
                # Assign decrements to latest possible free slots before deadline
                for j in range(deadline-1-1, -1, -1):
                    if j not in reserved and slots[j] < 1:
                        slots[j] = 1
                        req -= 1
                        if req == 0:
                            break
                if req > 0:
                    return False
            # Check if total decrements fit
            if sum(nums) > S - n:
                return False
            return True

        # Binary search
        left = 1
        right = m
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if can_mark(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        return result