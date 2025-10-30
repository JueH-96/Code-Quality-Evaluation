class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        n = len(batteryPercentages)
        low = 0
        high = n - 1
        while low < high:
            mid = (low + high + 1) // 2
            if batteryPercentages[mid] > mid:
                low = mid
            else:
                high = mid - 1
                
        return high + 1