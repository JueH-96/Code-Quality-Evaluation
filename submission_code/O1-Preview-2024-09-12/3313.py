class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        used = [False] * n
        selected_sums = []

        for _ in range(k):
            max_sum = float('-inf')
            min_sum = float('inf')
            curr_max_sum = 0
            curr_min_sum = 0
            max_start = max_end = -1
            min_start = min_end = -1
            temp_max_start = temp_min_start = 0

            for i in range(n):
                if used[i]:
                    curr_max_sum = 0
                    curr_min_sum = 0
                    temp_max_start = temp_min_start = i + 1
                    continue

                # Max sum subarray
                curr_max_sum += nums[i]
                if curr_max_sum < nums[i]:
                    curr_max_sum = nums[i]
                    temp_max_start = i
                if curr_max_sum > max_sum:
                    max_sum = curr_max_sum
                    max_start = temp_max_start
                    max_end = i

                # Min sum subarray
                curr_min_sum += nums[i]
                if curr_min_sum > nums[i]:
                    curr_min_sum = nums[i]
                    temp_min_start = i
                if curr_min_sum < min_sum:
                    min_sum = curr_min_sum
                    min_start = temp_min_start
                    min_end = i

            if max_sum == float('-inf') and min_sum == float('inf'):
                break  # No more subarrays

            # Compare abs(max_sum) and abs(min_sum)
            if abs(max_sum) >= abs(min_sum):
                # Select max_sum subarray
                selected_sums.append(max_sum)
                # Mark used positions
                for i in range(max_start, max_end + 1):
                    used[i] = True
            else:
                # Select min_sum subarray
                selected_sums.append(min_sum)
                # Mark used positions
                for i in range(min_start, min_end + 1):
                    used[i] = True

        # Compute the coefficients
        c = []
        for i in range(1, k + 1):
            sign = (-1) ** (i + 1)
            coeff = sign * (k - i + 1)
            c.append(coeff)

        # Now, compute total strength
        total_strength = 0
        for coeff, sum_i in zip(c, selected_sums):
            total_strength += coeff * sum_i

        return total_strength