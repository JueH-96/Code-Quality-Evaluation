from typing import List

class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        """
        Iterate through the devices once.
        `decrement` keeps track of how many times every subsequent device has
        been decreased so far. For the current device, its effective battery
        level is batteryPercentages[i] - decrement. If this value is positive,
        the device can be tested, we increase the answer and also increase
        `decrement` by 1 because every following device will be decreased by
        one more unit after this test.
        """
        tested = 0          # number of devices tested
        decrement = 0       # total decrement applied to the current index
        
        for charge in batteryPercentages:
            if charge - decrement > 0:
                tested += 1
                decrement += 1      # future devices will be reduced by 1 more
        return tested