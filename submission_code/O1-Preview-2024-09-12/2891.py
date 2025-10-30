class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for num in nums:
            l = num - k
            r = num + k
            events.append((l, 1))       # Interval start
            events.append((r + 1, -1))  # Interval end + 1
        events.sort()
        count = 0
        max_count = 0
        for point, delta in events:
            count += delta
            if count > max_count:
                max_count = count
        return max_count