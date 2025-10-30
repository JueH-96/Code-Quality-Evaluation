from typing import List

class Solution:
  def countTestedDevices(self, batteryPercentages: List[int]) -> int:
    
    tested_devices_count = 0
    
    # accumulated_decrement represents the total reduction effect on a device's battery
    # due to all prior devices that were tested.
    # If 'k' devices before the current one (say, device 'i') were tested, 
    # then device 'i's battery (and all subsequent ones) effectively lose 'k' percentage points
    # from their initial values by the time it's their turn to be evaluated.
    accumulated_decrement = 0
    
    # Iterate through each device by its index.
    # batteryPercentages[i] in this loop refers to the *initial* battery percentage of device i.
    for i in range(len(batteryPercentages)):
      
      # Calculate the effective battery percentage for device 'i' at the moment of its test.
      # This is its initial percentage minus the total drain from previously tested devices.
      effective_percentage = batteryPercentages[i] - accumulated_decrement
      
      # According to the problem: "If batteryPercentages[i] is greater than 0:"
      # This 'batteryPercentages[i]' in the problem description refers to the *current*
      # state of the battery after previous operations. 'effective_percentage' represents this value.
      if effective_percentage > 0:
        # Device 'i' has enough battery to be tested.
        tested_devices_count += 1
        
        # Since device 'i' is tested, it contributes to draining subsequent devices.
        # So, increment the 'accumulated_decrement' that will apply to devices j > i.
        accumulated_decrement += 1
        
    return tested_devices_count