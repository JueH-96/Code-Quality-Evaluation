class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        from collections import defaultdict

        n = len(nums)
        unique = sorted(list(set(nums)))
        m = len(unique)
        total = 0

        for i in range(m):
            a = unique[i]
            for j in range(i + 1, m):
                b = unique[j]
                if b - a <= 1:
                    continue

                # Check if both a and b exist in nums
                has_a = False
                has_b = False
                for num in nums:
                    if num == a:
                        has_a = True
                    if num == b:
                        has_b = True
                    if has_a and has_b:
                        break
                if not (has_a and has_b):
                    continue

                # Find allowed intervals
                intervals = []
                current_start = None
                for idx in range(n):
                    x = nums[idx]
                    if x <= a or x >= b:
                        if current_start is None:
                            current_start = idx
                    else:
                        if current_start is not None:
                            intervals.append((current_start, idx - 1))
                            current_start = None
                if current_start is not None:
                    intervals.append((current_start, n - 1))

                contribution = 0
                for (s, e) in intervals:
                    # Calculate total subarrays in the interval
                    length = e - s + 1
                    total_subarrays = length * (length + 1) // 2

                    # Calculate subarrays with no a
                    subarrays_no_a = 0
                    run_length = 0
                    for i in range(s, e + 1):
                        if nums[i] != a:
                            run_length += 1
                        else:
                            subarrays_no_a += run_length * (run_length + 1) // 2
                            run_length = 0
                    subarrays_no_a += run_length * (run_length + 1) // 2

                    # Calculate subarrays with no b
                    subarrays_no_b = 0
                    run_length = 0
                    for i in range(s, e + 1):
                        if nums[i] != b:
                            run_length += 1
                        else:
                            subarrays_no_b += run_length * (run_length + 1) // 2
                            run_length = 0
                    subarrays_no_b += run_length * (run_length + 1) // 2

                    # Calculate subarrays with neither a nor b
                    subarrays_no_ab = 0
                    run_length = 0
                    for i in range(s, e + 1):
                        if nums[i] != a and nums[i] != b:
                            run_length += 1
                        else:
                            subarrays_no_ab += run_length * (run_length + 1) // 2
                            run_length = 0
                    subarrays_no_ab += run_length * (run_length + 1) // 2

                    # Compute current contribution
                    current_contribution = total_subarrays - subarrays_no_a - subarrays_no_b + subarrays_no_ab
                    contribution += current_contribution

                total += contribution

        return total