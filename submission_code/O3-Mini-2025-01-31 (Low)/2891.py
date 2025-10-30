class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for num in nums:
            # Start of interval
            events.append((num - k, 1))
            # End of interval, note: use num + k + 1 for exclusive interval ending
            events.append((num + k + 1, -1))
            
        # Sort the events by the position
        events.sort()
        
        # Sweep line
        current = 0
        max_beauty = 0
        for pos, delta in events:
            current += delta
            if current > max_beauty:
                max_beauty = current
        return max_beauty