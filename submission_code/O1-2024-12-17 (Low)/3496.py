class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        import math
        
        # A function to check if we can reduce the mountain to height 0
        # within a given time limit T.
        def canFinishWithin(T: int) -> bool:
            total_reduced = 0
            for w in workerTimes:
                # Maximum x such that w * (x*(x+1)//2) <= T
                # => x*(x+1)/2 <= T / w
                # Solve x^2 + x - 2*(T/w) <= 0
                # x = floor( (-1 + sqrt(1 + 8*(T/w))) / 2 )
                # Make sure T>=w to avoid zero/negative inside sqrt
                if w > 0:
                    possible = T // w  # quick upper bound
                    # Then refine using the quadratic formula approach:
                    val = (1 + 8 * (T / w))
                    if val > 0:
                        x = int((-1 + math.isqrt(int(val))) // 2)
                        # The above uses integer sqrt for safety
                        # We might check if x is an overestimate or underestimate
                        while x*(x+1)//2 > possible:
                            x -= 1
                        while (x+1)*(x+2)//2 <= possible:
                            x += 1
                        total_reduced += x
                if total_reduced >= mountainHeight:
                    return True
            return total_reduced >= mountainHeight
        
        # Edge case: if mountainHeight == 0, no time needed.
        if mountainHeight == 0:
            return 0
        
        # Upper bound: if only one worker with the slowest time had to do all
        # That time = max(workerTimes) * (mountainHeight*(mountainHeight+1)//2)
        # But to keep it simpler, we can use that as a safe upper bound.
        max_w = max(workerTimes)
        left, right = 0, max_w * (mountainHeight*(mountainHeight+1)//2)
        
        # Binary search to find the minimum T such that canFinishWithin(T) is True
        while left < right:
            mid = (left + right) // 2
            if canFinishWithin(mid):
                right = mid
            else:
                left = mid + 1
        
        return left