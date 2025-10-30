from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        unique = sorted(set(nums))
        total = 0
        n = len(nums)
        for i in range(len(unique)):
            a = unique[i]
            for j in range(i + 1, len(unique)):
                b = unique[j]
                if b - a <= 1:
                    continue
                # Create the mask for this pair (a, b)
                mask = [1 if x <= a or x >= b else 0 for x in nums]
                # Find all intervals of consecutive 1s in the mask
                intervals = []
                start = 0
                while start < n:
                    if mask[start] == 1:
                        end = start
                        while end < n and mask[end] == 1:
                            end += 1
                        intervals.append((start, end - 1))
                        start = end
                    else:
                        start += 1
                # Process each interval
                for (s, e) in intervals:
                    # Check if interval contains at least one a and one b
                    has_a = False
                    has_b = False
                    for x in nums[s:e + 1]:
                        if x == a:
                            has_a = True
                        elif x == b:
                            has_b = True
                        if has_a and has_b:
                            break
                    if not (has_a and has_b):
                        continue
                    # Compute total_subarrays
                    L = e - s + 1
                    total_subarrays = L * (L + 1) // 2
                    # Compute subarrays_without_a
                    subarrays_without_a = 0
                    current = 0
                    for x in nums[s:e + 1]:
                        if x != a:
                            current += 1
                        else:
                            subarrays_without_a += current * (current + 1) // 2
                            current = 0
                    subarrays_without_a += current * (current + 1) // 2
                    # Compute subarrays_without_b
                    subarrays_without_b = 0
                    current = 0
                    for x in nums[s:e + 1]:
                        if x != b:
                            current += 1
                        else:
                            subarrays_without_b += current * (current + 1) // 2
                            current = 0
                    subarrays_without_b += current * (current + 1) // 2
                    # Compute subarrays_without_a_and_b
                    subarrays_without_ab = 0
                    current = 0
                    for x in nums[s:e + 1]:
                        if x != a and x != b:
                            current += 1
                        else:
                            subarrays_without_ab += current * (current + 1) // 2
                            current = 0
                    subarrays_without_ab += current * (current + 1) // 2
                    # Calculate the count using inclusion-exclusion
                    count = total_subarrays - subarrays_without_a - subarrays_without_b + subarrays_without_ab
                    total += count
        return total