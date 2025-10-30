class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for num in nums:
            events.append((num - k, 1))  # start of range
            events.append((num + k + 1, -1))  # end of range + 1
        
        events.sort()
        
        count = 0
        max_beauty = 0
        
        for _, event in events:
            count += event
            max_beauty = max(max_beauty, count)
        
        return max_beauty