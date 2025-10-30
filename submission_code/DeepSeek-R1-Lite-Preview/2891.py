class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # Create events for interval starts and ends
        events = []
        for num in nums:
            start = num - k
            end = num + k + 1  # end is exclusive
            events.append((start, 0))  # 0 indicates start
            events.append((end, 1))    # 1 indicates end
        
        # Sort the events by position; within the same position, starts come before ends
        events.sort()
        
        current_overlap = 0
        max_overlap = 0
        for event in events:
            if event[1] == 0:
                current_overlap += 1
                if current_overlap > max_overlap:
                    max_overlap = current_overlap
            else:
                current_overlap -= 1
        return max_overlap