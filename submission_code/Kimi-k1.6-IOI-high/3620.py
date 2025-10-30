class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        prev = -float('inf')
        count = 0
        for num in nums:
            current_min = num - k
            current_max = num + k
            if current_min > prev:
                prev = current_min
                count += 1
            else:
                desired = max(prev + 1, current_min)
                if desired <= current_max:
                    prev = desired
                    count += 1
                elif current_max > prev:
                    prev = current_max
                    count += 1
        return count