class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # We'll use a sweep-line algorithm to find the point on the number line
        # that lies in the largest number of intervals [nums[i]-k, nums[i]+k].
        # That largest intersection count is the answer.

        events = []
        
        for x in nums:
            left = x - k
            right = x + k
            # Start of interval
            events.append((left, 1))
            # One position past the end of interval
            events.append((right + 1, -1))
        
        # Sort events by coordinate
        events.sort(key=lambda e: e[0])
        
        curr = 0
        res = 0
        
        # Sweep through the coordinates
        for _, val in events:
            curr += val
            res = max(res, curr)
        
        return res