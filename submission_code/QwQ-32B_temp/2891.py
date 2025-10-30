class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        events = []
        for num in nums:
            s = num - k
            e = num + k
            events.append((s, 1))
            events.append((e + 1, -1))
        
        events.sort()
        max_count = 0
        current = 0
        for x, delta in events:
            current += delta
            if current > max_count:
                max_count = current
        return max_count