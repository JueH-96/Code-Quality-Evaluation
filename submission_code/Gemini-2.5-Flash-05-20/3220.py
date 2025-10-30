from typing import List

class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        # Initialize a counter for the number of devices that will be tested.
        # This variable also implicitly tracks the cumulative reduction applied
        # to the battery percentages of subsequent devices.
        tested_devices_count = 0 
        
        # Iterate through each device's initial battery percentage.
        # The iteration order (0 to n-1) is crucial as specified.
        for i in range(len(batteryPercentages)):
            # Calculate the effective battery percentage for the current device.
            # This is its initial percentage minus the number of devices that were
            # previously tested (which cause a -1 reduction for current and subsequent devices).
            effective_battery = batteryPercentages[i] - tested_devices_count
            
            # If the effective battery percentage is greater than 0, the device is tested.
            # A value of 0 or less means the battery is depleted or would be depleted,
            # so the device cannot be tested.
            if effective_battery > 0:
                # Increment the count of tested devices.
                # This increment also means that all devices with indices j > i
                # will experience an additional -1 reduction.
                tested_devices_count += 1
                
        # After checking all devices, return the total count of devices that were tested.
        return tested_devices_count