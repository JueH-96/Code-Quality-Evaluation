class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # We can treat each element nums[i] as an interval [nums[i] - k, nums[i] + k].
        # We want to find the maximum number of these intervals that overlap at some point,
        # because that point can be the chosen value to which we set all those elements.
        
        events = []
        for x in nums:
            left, right = x - k, x + k
            # Mark the start of the interval with +1, the end with -1
            events.append((left, 1))
            events.append((right, -1))
        
        # Sort events by coordinate; ties broken so that +1 is processed before -1
        events.sort(key=lambda e: (e[0], -e[1]))
        
        cover = 0
        max_cover = 0
        # Sweep line: add 1 when we enter an interval, subtract 1 when we leave
        for _, t in events:
            cover += t
            max_cover = max(max_cover, cover)
        
        return max_cover