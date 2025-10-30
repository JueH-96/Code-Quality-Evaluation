from typing import List

class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        """
        Counts the number of devices tested based on the given process.
        
        The process involves iterating through devices from 0 to n-1.
        If a device's battery percentage is positive, it is tested.
        Testing a device reduces the battery percentage of all subsequent
        devices by 1 (minimum 0).
        
        Args:
            batteryPercentages: A 0-indexed integer array representing battery
                                percentages of devices.
                                
        Returns:
            The total number of devices tested.
        """
        # tested_count tracks the total number of devices tested so far.
        # Crucially, this count also represents the total reduction applied
        # to the battery percentage of devices processed *after* the currently
        # considered one. Each device tested contributes a reduction of 1
        # to all subsequent devices' battery percentages.
        tested_count = 0
        
        # Iterate through the devices in order from index 0 to n - 1.
        for percentage in batteryPercentages:
            # Calculate the "effective" battery percentage for the current device.
            # This is the original percentage minus the cumulative reduction
            # applied by devices tested *before* this one. The cumulative reduction
            # amount is equal to the number of devices already tested (`tested_count`).
            effective_percentage = percentage - tested_count
            
            # If the effective percentage is greater than 0, it means the device
            # still has charge remaining after accounting for prior tests.
            # According to the rules, this device will be tested.
            if effective_percentage > 0:
                # Increment the total count of tested devices.
                # Testing this device adds one more to the count of devices
                # that reduce the battery percentage of *subsequent* devices.
                # So, the `tested_count` variable correctly tracks both the
                # total count of tested devices and the cumulative reduction
                # applied to following devices.
                tested_count += 1
                
        # After processing all devices, `tested_count` holds the final number
        # of devices that were tested.
        return tested_count