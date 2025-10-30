class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # Create events for line sweep
        events = []
        for num in nums:
            events.append((num - k, 1))  # Opening event
            events.append((num + k + 1, -1))  # Closing event
        
        events.sort()  # Sort events by position
        
        max_overlap = 0
        current_overlap = 0
        
        i = 0
        while i < len(events):
            pos = events[i][0]
            # Process all events at the current position
            while i < len(events) and events[i][0] == pos:
                current_overlap += events[i][1]
                i += 1
            
            # Update maximum overlap after processing all events at this position
            max_overlap = max(max_overlap, current_overlap)
        
        return max_overlap