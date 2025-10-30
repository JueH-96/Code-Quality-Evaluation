class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        # We maintain offset = original[i] - original[0] as we scan,
        # so that copy[i] = x + offset must lie in bounds[i].
        # That gives bounds[i][0] - offset <= x <= bounds[i][1] - offset.
        # We intersect all these intervals to find possible x values.
        
        n = len(original)
        # Initialize intersection to a very wide interval
        LOW = -10**18
        HIGH = 10**18
        low, high = LOW, HIGH
        
        offset = 0  # original[0] - original[0] == 0
        for i in range(n):
            # current allowed interval for x = copy[0]
            b0, b1 = bounds[i]
            cur_low = b0 - offset
            cur_high = b1 - offset
            # intersect
            if cur_low > low:
                low = cur_low
            if cur_high < high:
                high = cur_high
            # If empty, no solutions
            if low > high:
                return 0
            # update offset for next i
            if i + 1 < n:
                offset += original[i+1] - original[i]
        
        # Number of integer x in [low, high]
        return max(0, high - low + 1)