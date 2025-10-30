class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for num in nums:
            start = num - k
            end = num + k + 1  # end is exclusive
            events.append((start, 1))
            events.append((end, -1))
        
        # Sort the events. For same positions, -1 events come before 1 events due to their natural order.
        events.sort()
        
        max_count = 0
        current = 0
        for pos, delta in events:
            current += delta
            if current > max_count:
                max_count = current
        return max_count