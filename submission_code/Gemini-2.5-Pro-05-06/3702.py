from typing import List

class Solution:
  # Helper function to check the pairwise coprime condition for arrays of length >= 3
  def _are_pairwise_coprime(self, sub_array: List[int]) -> bool:
      # The numbers in sub_array are pairwise coprime if for each prime p,
      # p divides at most one number in sub_array (ignoring 1s).
      # Relevant primes for numbers up to 10 are 2, 3, 5, 7.
      primes = [2, 3, 5, 7]
      
      for p in primes:
          count_divisible_by_p = 0
          for num in sub_array:
              # Numbers equal to 1 are coprime with everything and do not have prime factors.
              if num > 1 and num % p == 0:
                  count_divisible_by_p += 1
          
          if count_divisible_by_p > 1:
              # This prime p divides more than one number (that is > 1)
              # in the sub_array. This means not all numbers are pairwise coprime.
              return False
      return True

  def maxLength(self, nums: List[int]) -> int:
      n = len(nums)
      
      # Constraints: 2 <= nums.length <= 100, 1 <= nums[i] <= 10.
      # So, n is always at least 2.
      
      max_len = 0
      
      # Iterate over all possible subarrays nums[i..j]
      for i in range(n):
          for j in range(i, n):
              current_subarray = nums[i : j+1] # Python slice creates a copy
              L = len(current_subarray)
              
              is_product_equivalent = False # Flag
              
              if L == 1:
                  # For a single element array [x]:
                  # prod([x]) = x, gcd([x]) = x, lcm([x]) = x
                  # Condition: x == x * x  => x^2 - x = 0 => x(x-1) = 0.
                  # Since nums[i] are positive integers (>=1), x must be 1.
                  if current_subarray[0] == 1:
                      is_product_equivalent = True
              elif L == 2:
                  # For a two-element array [a, b]:
                  # prod([a,b]) = a*b
                  # It's a standard property that for any two positive integers a, b:
                  # a * b = gcd(a,b) * lcm(a,b)
                  # So, any array of length 2 is product equivalent.
                  is_product_equivalent = True
              else: # L >= 3
                  # For an array arr of length k >= 3, it's product equivalent iff
                  # its elements are pairwise coprime. (Derived from prime exponent analysis).
                  if self._are_pairwise_coprime(current_subarray):
                      is_product_equivalent = True
              
              if is_product_equivalent:
                  if L > max_len: 
                      max_len = L
                      
      return max_len