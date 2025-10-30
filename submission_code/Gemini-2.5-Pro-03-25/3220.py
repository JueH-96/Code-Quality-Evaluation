import math # The problem statement's template might include math, though it's not used here.
from typing import List # Required for type hinting the input list.

class Solution:
  """
  Implements the solution to count tested devices based on battery percentages
  following a specific testing protocol.
  """
  def countTestedDevices(self, batteryPercentages: List[int]) -> int:
    """
    Calculates the number of devices that will be tested according to the rules.

    The testing process involves iterating through devices sequentially. 
    If a device `i` has a battery percentage greater than 0 at the time it's checked,
    it is counted as tested. Testing device `i` reduces the battery percentage 
    of all subsequent devices `j > i` by 1 (but not below 0).

    This method uses an optimized O(n) approach. It keeps track of the number 
    of devices tested so far (`tested_count`). This count also represents the 
    total reduction that would have been applied to the current device's 
    battery percentage due to tests of preceding devices. A device `i` is tested 
    if its initial battery percentage `batteryPercentages[i]` is greater than 
    the `tested_count` accumulated before checking device `i`.

    Args:
      batteryPercentages: A 0-indexed list of integers representing the initial 
                          battery percentages of the devices.

    Returns:
      An integer representing the total number of devices that will be tested.
    """
    
    # Initialize 'tested_count' to 0. This variable tracks the number of 
    # devices that have been successfully tested according to the rules.
    # It also implicitly represents the cumulative reduction in battery percentage 
    # that would have been applied to the current device being considered, 
    # due to the testing of devices before it.
    tested_count = 0
    
    # Get the total number of devices.
    n = len(batteryPercentages)
    
    # Iterate through each device using its index 'i' from 0 to n-1.
    for i in range(n):
      # For the current device `i`, its initial battery percentage is batteryPercentages[i].
      # The number of devices tested *before* this one is currently stored in `tested_count`.
      # Each previously tested device would effectively reduce the current device's 
      # battery percentage by 1.
      # So, the effective battery percentage at the moment of checking device `i` 
      # would be `batteryPercentages[i] - tested_count`.
      # The device `i` is tested if this effective percentage is greater than 0.
      # This condition is mathematically equivalent to checking if the initial 
      # battery percentage `batteryPercentages[i]` is strictly greater than 
      # the count of devices tested so far (`tested_count`).
      if batteryPercentages[i] > tested_count:
          # If the condition holds true, it means device `i` has enough battery
          # charge remaining (> 0) after accounting for the drain from previous tests.
          # Therefore, device `i` is tested.
          # Increment `tested_count` because we have now tested one more device.
          # This updated `tested_count` will be used to calculate the effective 
          # battery percentage for the subsequent devices.
          tested_count += 1
          
    # After iterating through all the devices, 'tested_count' holds the
    # final total number of devices that were tested.
    return tested_count