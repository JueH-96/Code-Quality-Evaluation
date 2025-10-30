from typing import List

class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        """
        Each time we test a device i (batteryPercentages[i] > 0), we effectively
        drain 1% from every subsequent device.  So by the time we reach index i,
        it has been drained exactly `count` times, where count is the number of
        tests performed on indices < i.  Hence batteryPercentages[i] is still > 0
        iff batteryPercentages[i] > count.
        """
        count = 0
        for bp in batteryPercentages:
            if bp > count:
                count += 1
        return count