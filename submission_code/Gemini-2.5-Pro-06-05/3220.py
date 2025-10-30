from typing import List

class Solution:
  """
  A class to solve the count tested devices problem.
  """
  def countTestedDevices(self, batteryPercentages: List[int]) -> int:
    """
    Counts the number of devices that will be tested based on a series of operations.

    Args:
      batteryPercentages: A list of integers representing the initial battery
                        percentages of devices.

    Returns:
      An integer denoting the total number of devices that will be tested.
    """
    
    # This variable tracks the number of devices tested so far.
    # It also represents the cumulative battery drain on subsequent devices.
    tested_devices_count = 0
    
    for battery in batteryPercentages:
      # A device is tested if its current battery percentage is greater than 0.
      # The current percentage is its initial percentage minus the number of
      # devices tested before it.
      # So, the condition is: battery - tested_devices_count > 0
      if battery > tested_devices_count:
        # If the condition holds, this device is tested.
        # We increment the count, which also increases the drain for
        # all subsequent devices.
        tested_devices_count += 1
        
    return tested_devices_count