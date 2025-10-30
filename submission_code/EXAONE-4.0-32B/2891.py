class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for num in nums:
            start = num - k
            end_plus1 = num + k + 1
            events.append((start, 1))
            events.append((end_plus1, -1))
        
        events.sort()
        
        current_count = 0
        max_count = 0
        for coord, delta in events:
            current_count += delta
            if current_count > max_count:
                max_count = current_count
                
        return max_count