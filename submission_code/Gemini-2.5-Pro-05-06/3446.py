from typing import List

class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        good_pairs_count = 0
        
        # Get lengths of the arrays, though not strictly needed if using enhanced for loops
        # n = len(nums1)
        # m = len(nums2)
        
        # Iterate through each element num1_val from nums1 (corresponds to nums1[i])
        for num1_val in nums1:
            # Iterate through each element num2_val from nums2 (corresponds to nums2[j])
            for num2_val in nums2:
                # Calculate the divisor term: nums2[j] * k
                # Constraints state:
                # 1 <= nums2[j] <= 50
                # 1 <= k <= 50
                # So, num2_val >= 1 and k >= 1.
                # This means divisor_term will always be positive (>= 1).
                divisor_term = num2_val * k
                
                # Check the divisibility condition: nums1[i] is divisible by (nums2[j] * k)
                # This translates to num1_val % divisor_term == 0.
                
                # Constraints state:
                # 1 <= nums1[i] <= 50
                # So, num1_val >= 1.
                
                # If num1_val < divisor_term, then num1_val % divisor_term will be num1_val itself.
                # Since num1_val >= 1, num1_val % divisor_term will not be 0 in this case.
                # So, the condition num1_val % divisor_term == 0 implicitly handles
                # cases where num1_val < divisor_term, correctly identifying them as not divisible.
                # Example: num1_val=5, divisor_term=10. 5 % 10 = 5. 5 != 0. Not a good pair.
                
                if num1_val % divisor_term == 0:
                    good_pairs_count += 1
                    
        return good_pairs_count