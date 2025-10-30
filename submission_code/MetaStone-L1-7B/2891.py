class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for num in nums:
            a = num - k
            b = num + k
            events.append((a, 0))  # Start event
            events.append((b + 1, 1))  # End event
        # Sort the events by position, and for the same position, start events come before end events
        events.sort(key=lambda x: (x[0], x[1]))
        current_count = 0
        max_count = 0
        for pos, typ in events:
            if typ == 0:
                current_count += 1
            else:
                current_count -= 1
            if current_count > max_count:
                max_count = current_count
        return max_count