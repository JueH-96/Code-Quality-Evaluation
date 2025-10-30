class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        workerTimes.sort()
        n = len(workerTimes)
        left = 0
        right = 10**18  # A sufficiently large upper bound
        answer = right
        
        while left <= right:
            mid = (left + right) // 2
            total = 0
            for t in workerTimes:
                # Binary search to find the maximum x where t*x*(x+1)/2 <= mid
                low_x, high_x = 0, mountainHeight
                best_x = 0
                while low_x <= high_x:
                    mid_x = (low_x + high_x) // 2
                    time_needed = t * mid_x * (mid_x + 1) // 2
                    if time_needed <= mid:
                        best_x = mid_x
                        low_x = mid_x + 1
                    else:
                        high_x = mid_x - 1
                total += best_x
                if total > mountainHeight:
                    break  # Early exit if sum exceeds mountainHeight
            if total >= mountainHeight:
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        return answer