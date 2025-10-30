from typing import List # For type hinting List[int]

class Solution:
  def countCompleteDayPairs(self, hours: List[int]) -> int:
    # We need to find pairs (hours[i], hours[j]) such that i < j
    # and (hours[i] + hours[j]) % 24 == 0.

    # This condition is equivalent to (hours[i] % 24 + hours[j] % 24) % 24 == 0.
    # Let rem_i = hours[i] % 24 and rem_j = hours[j] % 24.
    # We need (rem_i + rem_j) % 24 == 0.
    # This means rem_i + rem_j is either 0 or 24.
    # (Since 0 <= rem_i < 24 and 0 <= rem_j < 24, their sum is in [0, 46]).

    # For each hours[j] (let's call it current_hour_val), we calculate its remainder current_rem.
    # We need to find a previous hours[i] (let's call it prev_hour_val)
    # with remainder prev_rem such that:
    # if current_rem == 0, then prev_rem == 0.
    # if current_rem != 0, then prev_rem == 24 - current_rem.
    # This can be summarized as: prev_rem = (24 - current_rem) % 24.

    # We can use an array to store counts of remainders encountered so far.
    # remainder_counts[k] will store the number of times a remainder k has been seen
    # from elements processed *before* the current one.
    
    # Initialize an array of size 24 to store frequencies of remainders (0 to 23).
    # All counts are initially 0.
    remainder_counts = [0] * 24
    
    # Initialize the count of valid pairs to 0.
    pair_count = 0

    # Iterate through each hour value in the input list.
    for current_hour_val in hours:
        current_rem = current_hour_val % 24
        
        # Determine the remainder we need from a previous element
        # to make the sum of current_hour_val and prev_hour_val a multiple of 24.
        # If current_rem is r, we need a previous element with remainder (24-r)%24.
        # Example: If current_rem is 10, complement_rem is (24-10)%24 = 14.
        # Example: If current_rem is 0 (e.g. current_hour_val = 24 or 48),
        #          complement_rem is (24-0)%24 = 0.
        complement_rem = (24 - current_rem) % 24
        
        # Add the number of times we've seen the complement_rem among previous elements.
        # Each such previously seen element forms a valid pair with current_hour_val.
        # This correctly handles the i < j condition because we are pairing the
        # current element with elements already processed and recorded in remainder_counts.
        pair_count += remainder_counts[complement_rem]
        
        # Increment the frequency of the current_rem, as we've now processed current_hour_val.
        # This makes it available for subsequent elements to form pairs with.
        remainder_counts[current_rem] += 1
            
    return pair_count