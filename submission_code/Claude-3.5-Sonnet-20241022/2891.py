class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        n = len(nums)
        events = []
        
        # For each number, create an interval [num-k, num+k]
        # Store the start and end points as events
        for num in nums:
            events.append((num - k, 1))  # 1 represents start of interval
            events.append((num + k + 1, -1))  # -1 represents end of interval
            
        # Sort events by position
        events.sort()
        
        curr = 0  # Current overlapping intervals
        max_beauty = 0
        
        # Sweep line algorithm
        for _, change in events:
            curr += change
            max_beauty = max(max_beauty, curr)
            
        return max_beauty