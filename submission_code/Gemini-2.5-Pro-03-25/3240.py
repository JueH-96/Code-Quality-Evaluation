import math

class Solution:
  """
  Finds the maximum integer `num` such that the sum of prices of numbers from 1 to `num` is less than or equal to k.
  The price of a number `i` is defined based on its binary representation: it's the count of indices `p` such that the (p*x)-th bit is 1.
  The bits are 1-indexed from right to left.
  The total sum is S(num) = sum_{i=1 to num} Price(i). We seek the maximum `num` such that S(num) <= k.
  """
  def findMaximumNumber(self, k: int, x: int) -> int:
    
    def countSetBits(num, j):
        """
        Calculates C(num, j): the count of numbers `i` in the range [1, num] such that the j-th bit of `i` is 1.
        
        Uses the property that the j-th bit pattern repeats every 2^j numbers.
        In each block of 2^j numbers, the j-th bit is 1 for exactly 2^(j-1) numbers.
        
        We use the N = num + 1 formulation to count set bits in the range [0, N-1], which is equivalent to [0, num].
        Since Price(0) = 0, this count is equivalent to the count for [1, num].
        The j-th bit corresponds to the value 2^(j-1).
        """
        # j is the 1-indexed bit position. Check if j is valid.
        if j <= 0: 
            # This case should not normally be reached given x >= 1.
            return 0 
        
        # Calculate using N = num + 1. The range is [0, N-1].
        N = num + 1
        
        # Calculate powers of 2 needed using bit shifts for efficiency.
        # Python handles arbitrary precision integers, so standard integers limits are not an issue.
        power_of_2_j = 1 << j          # Computes 2^j
        power_of_2_j_minus_1 = 1 << (j - 1)  # Computes 2^(j-1)
        
        # Calculate the number of full blocks of size 2^j within the range [0, N-1].
        q = N // power_of_2_j          
        
        # Calculate the size of the remaining partial block at the end: [q * 2^j, N-1].
        r = N % power_of_2_j          
        
        # Calculate the total count.
        # Contribution from full blocks: q blocks * 2^(j-1) set bits per block.
        count = q * power_of_2_j_minus_1
        
        # Contribution from the partial block:
        # Numbers in the partial block are i = q*2^j + rem, where 0 <= rem < r.
        # The j-th bit is set if rem >= 2^(j-1).
        # We need to count 'rem' such that 2^(j-1) <= rem < r.
        # The number of such 'rem' is max(0, r - 2^(j-1)).
        count += max(0, r - power_of_2_j_minus_1)
        
        return count

    def calculate_S(num, x):
        """
        Calculates the total sum of prices S(num) = sum_{i=1 to num} Price(i).
        Price(i) = sum_{p=1}^\infty [ (p*x)-th bit of i is 1 ]
        S(num) = sum_{i=1 to num} sum_{p=1}^\infty [ (p*x)-th bit of i is 1 ]
               = sum_{p=1}^\infty sum_{i=1 to num} [ (p*x)-th bit of i is 1 ]
               = sum_{p=1}^\infty C(num, p*x)
        """
        # Base case: if num is 0, the sum is 0.
        if num == 0:
            return 0
            
        total_price_sum = 0
        j = x # Start checking bit positions from x.
        while True:
            # Calculate the value represented by the j-th bit, which is 2^(j-1).
            power_of_2_j_minus_1 = 1 << (j - 1)
            
            # Optimization: If the value 2^(j-1) is already greater than num,
            # then for any number i <= num, the j-th bit (and any higher bit j' > j) 
            # must be 0. The contribution C(num, j') for all j' >= j will be 0.
            # Thus, we can stop the summation process.
            if power_of_2_j_minus_1 > num:
                 break

            # Calculate C(num, j): count of numbers in [1, num] with j-th bit set.
            current_count = countSetBits(num, j)
            
            # Add this count to the total sum.
            total_price_sum += current_count
            
            # Move to the next relevant bit position: j + x.
            j += x 
            
        return total_price_sum

    # Binary search for the maximum `num`.
    # The function calculate_S(num, x) is monotonically non-decreasing with `num`.
    # We need to find the largest `num` such that calculate_S(num, x) <= k.
    
    low = 0 # Start search from 0. The smallest possible answer is 0.
    # Set a sufficiently large upper bound for the search space.
    # Based on analysis, num could be up to the order of 10^14 to 10^15 for max k=10^15.
    # 2 * 10^15 is a safe upper bound. Larger bounds like 10^18 also work but might slightly increase iterations.
    high = 2 * 10**15  
    ans = 0 # Initialize the answer to 0.

    while low <= high:
        mid = (low + high) // 2
        
        # Calculate the sum of prices for numbers up to `mid`.
        current_sum = calculate_S(mid, x)
        
        if current_sum <= k:
            # If the sum is within the limit k, `mid` is a feasible solution.
            # We store `mid` as the best answer found so far and try to find a larger feasible `num`.
            ans = mid 
            low = mid + 1
        else:
            # If the sum exceeds k, `mid` is too large.
            # We need to search in the lower half.
            high = mid - 1
            
    # After the binary search loop terminates, `ans` holds the largest `num` found satisfying the condition.
    return ans