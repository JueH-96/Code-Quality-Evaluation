class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        s = sum(nums)
        min_length = float('inf')

        for length in range(1, 2 * n + 1):
            for start in range(length):
                current_sum = 0
                for i in range(start, start + length):
                    current_sum += nums[i % n]
                if current_sum == target:
                    min_length = min(min_length, length)
                    break
            if min_length != float('inf'):
                break

        if min_length != float('inf'):
            return min_length

        if s <= 0 and target > 0:
            return -1

        if target == 0:
            return 1 if 0 in nums else -1

        for num_copies in range(1, target // s + 2 if s > 0 else 3):
            remaining_target = target - num_copies * s

            if remaining_target == 0 and num_copies > 0:
                min_length = min(min_length, num_copies * n)
                break

            if n > 0:
                for start in range(n):
                    current_sum = 0
                    for end in range(start, n):
                        current_sum += nums[end]
                        if current_sum == remaining_target:
                            min_length = min(min_length, num_copies * n + (end - start + 1))
                            break
                        elif current_sum > remaining_target:
                            break

        if min_length != float('inf'):
            return min_length

        prefix_sums = [0] * (n + 1)
        for i in range(n):
            prefix_sums[i + 1] = prefix_sums[i] + nums[i]

        for length in range(1, n + 1):
            for i in range(n - length + 1):
                if prefix_sums[i + length] - prefix_sums[i] == target:
                    min_length = min(min_length, length)

        for length in range(n + 1, 2 * n + 1):
            for start in range(n):
                end = start + length - 1
                num_full_copies = (end // n) - (start // n)
                rem_start = start % n
                rem_end = end % n

                current_sum = 0
                for k in range(start, end + 1):
                    current_sum += nums[k % n]

                if current_sum == target:
                    min_length = min(min_length, length)

        if min_length != float('inf'):
            return min_length
        else:
            return -1