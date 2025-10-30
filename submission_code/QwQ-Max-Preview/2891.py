class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for x in nums:
            s = x - k
            e = x + k
            events.append((s, 1))   # start event (delta +1)
            events.append((e + 1, 0))  # end event (delta -1)
        
        events.sort()
        max_count = 0
        current_count = 0
        
        for pos, typ in events:
            if typ == 1:
                current_count += 1
                if current_count > max_count:
                    max_count = current_count
            else:
                current_count -= 1
        
        return max_count