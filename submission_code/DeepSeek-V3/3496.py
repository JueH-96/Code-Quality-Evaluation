class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        left, right = 1, 10**14
        while left < right:
            mid = (left + right) // 2
            total = 0
            for t in workerTimes:
                # Calculate the maximum x such that t * (x * (x + 1)) // 2 <= mid
                # Solving t * x * (x + 1) / 2 <= mid
                # x^2 + x - 2*mid/t <= 0
                # Using quadratic formula: x = [-1 + sqrt(1 + 8*mid/t)] / 2
                discriminant = 1 + 8 * mid / t
                x = int(( -1 + discriminant ** 0.5 ) // 2)
                total += x
                if total >= mountainHeight:
                    break
            if total >= mountainHeight:
                right = mid
            else:
                left = mid + 1
        return left