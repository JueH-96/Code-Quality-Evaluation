import math

class Solution:
  """
  Counts the number of "good" n-digit integers.
  An integer is good if its digits can be rearranged to form a k-palindromic integer.
  A k-palindromic integer is a palindrome divisible by k.
  Both the original integer and the rearranged palindrome must be n-digit numbers (no leading zeros).
  Constraints: 1 <= n <= 10, 1 <= k <= 9.
  """
  def countGoodIntegers(self, n: int, k: int) -> int:
      
      # Precompute factorials up to n for calculating permutations efficiently.
      # We need factorials from 0! to n!. The list should have size n + 1.
      factorials = [math.factorial(i) for i in range(n + 1)]

      def calculate_permutations(counts_tuple):
          """
          Calculates the number of unique n-digit integers that can be formed 
          by rearranging digits with the given counts in counts_tuple.
          An n-digit integer cannot have a leading zero.

          Args:
              counts_tuple: A tuple of length 10, where counts_tuple[i] is the count of digit i.
                            The sum of counts must equal n.

          Returns:
              The count of distinct n-digit integers formed by permutations of the digits.
          """
          
          c_0 = counts_tuple[0] # Count of digit 0

          # Calculate the total number of permutations of the multiset of digits.
          # This is given by the multinomial coefficient: P_M = n! / (c_0! * c_1! * ... * c_9!)
          denominator = 1
          for count in counts_tuple:
               # factorials[count] gives the factorial of the count of each digit.
               # The product forms the denominator of the multinomial coefficient.
               denominator *= factorials[count]
          
          # The denominator should mathematically always be positive if counts are non-negative integers.
          # A zero denominator would imply an issue like invalid counts or numerical errors (unlikely with Python's arbitrary precision integers).
          if denominator == 0:
               # This path should not be reached under normal circumstances.
               return 0 
          
          # Use integer division `//` as the result must be an integer.
          P_M = factorials[n] // denominator

          # If the count of the digit 0 (c_0) is 0, then no permutation can start with 0.
          # All P_M permutations result in valid n-digit integers.
          if c_0 == 0:
              return P_M
          else:
              # If the count of digit 0 (c_0) is greater than 0, some permutations will start with 0.
              # These permutations do not form valid n-digit integers. We need to subtract them.
              # The number of permutations starting with 0 is calculated by fixing 0 as the first digit
              # and permuting the remaining n-1 digits. The counts for the remaining digits are (c_0-1, c_1, ..., c_9).
              # The number of permutations starting with 0 is (n-1)! / ((c_0-1)! * c_1! * ... * c_9!).
              # A simpler way to compute the count of valid n-digit integers is using the derived formula:
              # Valid permutations count C_M = P_M * (n - c_0) / n.
              # To ensure integer arithmetic, compute as C_M = (n - c_0) * P_M // n.
              
              # Check for n=0 edge case, although constraints state n >= 1.
              if n == 0: 
                  return 0
              
              # Perform the calculation using integer division.
              # The result is guaranteed to be an integer because the number of permutations starting with 0 must be an integer.
              C_M = (n - c_0) * P_M // n
              return C_M

      # Use a set to store the unique multisets of digits found for k-palindromic numbers.
      # A multiset is represented by a tuple of counts for digits 0 through 9.
      good_multisets = set()
      
      # Calculate the length of the first half of the palindrome.
      # If n is even (n=2p), the first half has length p. Palindrome is d1..dp dp..d1.
      # If n is odd (n=2p+1), the first half has length p+1. Palindrome is d1..dp d(p+1) dp..d1.
      # The length m = ceil(n/2) = (n+1)//2 using integer division.
      m = (n + 1) // 2
      
      # Define the range for iterating through possible first halves.
      # The first digit (d1) of the palindrome must be non-zero (1-9).
      # The first half number must start with a non-zero digit.
      # If m=1 (n=1 or n=2), the first half is a single digit d1 from 1 to 9. Range [1, 10).
      # If m>1, the first half is an m-digit number. Smallest is 10**(m-1). Largest is 10**m - 1. Range [10**(m-1), 10**m).
      start = 10**(m - 1) if m > 1 else 1
      end = 10**m # range() is exclusive of the end value.
      
      # Iterate through all possible integer values representing the first half of the palindrome.
      for first_half_num in range(start, end):
          s_prefix = str(first_half_num) # Convert the first half number to string.
          
          # Construct the full palindrome string based on the first half string `s_prefix`.
          if n % 2 == 0: # n is even
              # The second half is the reverse of the first half.
              s_suffix = s_prefix[::-1]
              P_str = s_prefix + s_suffix
          else: # n is odd
              # The second half is the reverse of the first half excluding its last character (the middle digit).
              s_suffix = s_prefix[:-1][::-1] 
              P_str = s_prefix + s_suffix

          # Convert the palindrome string to an integer.
          x = int(P_str)
          
          # Check if the palindrome integer x is divisible by k.
          if x % k == 0:
              # If x is k-palindromic, calculate the multiset of its digits.
              counts = [0] * 10 # Initialize counts for digits 0-9.
              for digit_char in P_str:
                  counts[int(digit_char)] += 1
              
              # Add the tuple representation of the digit counts to the set.
              # The set automatically handles duplicates, ensuring each unique multiset is stored only once.
              good_multisets.add(tuple(counts))

      # Sum the number of valid n-digit permutations for each unique good multiset found.
      total_count = 0
      for counts_tuple in good_multisets:
          # For each multiset that can form a k-palindrome, calculate how many distinct n-digit numbers can be formed using its digits.
          total_count += calculate_permutations(counts_tuple)
            
      # Return the total count of good integers.
      return total_count