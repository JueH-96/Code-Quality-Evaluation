from typing import List

class Solution:
  def minBitwiseArray(self, nums: List[int]) -> List[int]:
    ans_list = []
    for num in nums:
      if num == 2:
        ans_list.append(-1)
        continue
      
      # num is an odd prime, so num >= 3.
      
      # Candidate 1: ans_val is even.
      # If ans_val is even, then (ans_val | (ans_val + 1)) == ans_val + 1.
      # So, we need ans_val + 1 == num. This means ans_val = num - 1.
      # Since num is an odd prime (num >= 3), num - 1 is an even integer (>= 2).
      # This is a valid candidate type.
      min_found_ans_val = num - 1
      
      # Candidate 2: ans_val is odd.
      # Let k be the number of trailing 1s in ans_val's binary representation.
      # Since ans_val is odd, k >= 1.
      # ans_val can be written as X01...1 (k ones), where X is some binary prefix.
      # ans_val + 1 will be X10...0 (k zeros).
      # Then (ans_val | (ans_val + 1)) == X11...1 (k ones).
      # Let this result be target_val. So target_val = num.
      # This means 'num' must have (k + 1) trailing ones in its binary representation.
      # And ans_val is 'num' with its k-th bit (0-indexed) flipped from 1 to 0.
      # So, ans_val = num - (1 << k).
      
      # We iterate through possible values of k.
      # k must be >= 1.
      current_k = 1
      while True:
        # power_of_2_k represents 2^current_k
        power_of_2_k = 1 << current_k
        
        # If 2^current_k >= num:
        #   If 2^current_k == num, then num - 2^current_k = 0. Candidate ans_val = 0.
        #     But 0 is even, and this case is for ans_val being odd. So 0 is not valid here.
        #   If 2^current_k > num, then num - 2^current_k < 0. Candidate ans_val is negative. Not valid.
        # So, we must have 2^current_k < num for a valid positive odd candidate.
        if power_of_2_k >= num:
            break
          
        # Check if num has (current_k + 1) trailing ones.
        # This means (num % 2^(current_k+1)) == (2^(current_k+1) - 1).
        # Or, (num & mask) == mask, where mask = (1 << (current_k + 1)) - 1.
        
        num_trailing_ones_mask = (1 << (current_k + 1)) - 1
        
        if (num & num_trailing_ones_mask) == num_trailing_ones_mask:
          # If num has (current_k + 1) trailing ones, then ans_val = num - 2^current_k is a candidate.
          candidate_ans_val = num - power_of_2_k
          
          # This candidate_ans_val is num - 2^current_k.
          # num is odd, 2^current_k is even (as current_k >= 1). So candidate_ans_val is odd.
          # It is also positive because power_of_2_k < num.
          # By construction, it has current_k trailing 1s and its (current_k)-th bit is 0.
          min_found_ans_val = min(min_found_ans_val, candidate_ans_val)
        
        # Max num is 1000. Max current_k is around 9 (2^9=512, 2^10=1024).
        # The loop will terminate because power_of_2_k grows exponentially.
        current_k += 1
            
      ans_list.append(min_found_ans_val)
            
    return ans_list