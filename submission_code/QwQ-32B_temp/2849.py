import bisect

class Solution:
    def sumImbalanceNumbers(self, nums):
        total = 0
        n = len(nums)
        for j in range(n):
            current = []
            current_imbalance = 0
            total += current_imbalance  # Add the subarray [j] which has 0 imbalance
            current.append(nums[j])
            for i in range(j-1, -1, -1):
                x = nums[i]
                pos = bisect.bisect_left(current, x)
                left = current[pos-1] if pos > 0 else None
                right = current[pos] if pos < len(current) else None

                prev_contribution = 0
                if left is not None and right is not None:
                    prev_contribution = 1 if (right - left > 1) else 0

                left_contribution = 0
                if left is not None:
                    left_contribution = 1 if (x - left > 1) else 0

                right_contribution = 0
                if right is not None:
                    right_contribution = 1 if (right - x > 1) else 0

                delta = (left_contribution + right_contribution) - prev_contribution
                current_imbalance += delta
                total += current_imbalance

                # Insert x into the current list
                current.insert(pos, x)
        return total