from typing import List

# Helper function outside the class
def get_exponent(n: int, p: int) -> int:
    """Calculates the exponent of prime p in the prime factorization of n."""
    count = 0
    # Constraint: 1 <= n <= 10
    # Simple division is efficient enough for small n.
    while n > 0 and n % p == 0:
        count += 1
        n //= p
    return count

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        n = len(nums)
        # The problem guarantees 2 <= nums.length, so there is always at least
        # one subarray of length 2, and any length 2 subarray is product equivalent.
        # Thus, the minimum possible answer is 2. Initializing max_len = 0
        # is safe as the loops will find a length 2 subarray and update it.
        max_len = 0

        # Primes relevant for numbers up to 10
        # We only care about prime factors present in numbers 1 through 10.
        # Primes are 2, 3, 5, 7.
        primes = [2, 3, 5, 7]

        # Precompute exponents for numbers 1 to 10 for each relevant prime.
        # This avoids recomputing exponents repeatedly inside the inner loops.
        # prime_exp[number][prime] = exponent
        prime_exp = {}
        for num in range(1, 11):
            prime_exp[num] = {}
            for p in primes:
                prime_exp[num][p] = get_exponent(num, p)

        # Iterate through all possible start indices i for subarrays.
        for i in range(n):
            # Initialize tracking variables for the subarray starting at index i.
            # These variables will store the sum, minimum, and maximum of exponents
            # for the current subarray nums[i:j+1] for each prime p.
            current_sum = {p: 0 for p in primes}
            current_min = {p: float('inf') for p in primes} # Initialize with infinity
            current_max = {p: float('-inf') for p in primes} # Initialize with negative infinity

            # Iterate through all possible end indices j (inclusive) starting from i.
            # This generates all subarrays starting at i.
            for j in range(i, n):
                # Get the element being added to the current subarray nums[i:j+1].
                current_element = nums[j]
                # Calculate the length of the current subarray.
                current_len = j - i + 1

                # Update sum, min, and max of exponents for each relevant prime p
                # based on the newly added element nums[j].
                for p in primes:
                    # Get the precomputed exponent for nums[j] and prime p.
                    exp = prime_exp[current_element][p]
                    
                    # Update the sum of exponents for prime p in the subarray nums[i:j+1].
                    current_sum[p] += exp
                    # Update the minimum exponent for prime p in the subarray nums[i:j+1].
                    current_min[p] = min(current_min[p], exp)
                    # Update the maximum exponent for prime p in the subarray nums[i:j+1].
                    current_max[p] = max(current_max[p], exp)

                # Check if the current subarray nums[i:j+1] is product equivalent.
                is_pe = False
                
                # Case 1: Subarray has length 1.
                # A subarray [a] is product equivalent if and only if a == 1.
                if current_len == 1:
                    if current_element == 1:
                        is_pe = True
                        
                # Case 2: Subarray has length 2.
                # Any subarray [a, b] of positive integers is always product equivalent
                # because prod(a,b) = a*b and lcm(a,b)*gcd(a,b) = a*b.
                elif current_len == 2:
                     is_pe = True
                     
                # Case 3: Subarray has length greater than 2.
                # A subarray [a1, ..., ak] with k > 2 is product equivalent if and only if,
                # for every relevant prime p, the sum of the exponents v_p(ai)
                # for all elements ai in the subarray equals the sum of the minimum
                # and maximum exponents v_p(ai) among all elements in the subarray.
                else: # current_len > 2
                    # Assume the subarray is PE until the condition fails for any prime.
                    is_pe = True 
                    for p in primes:
                        # Check the exponent condition: sum(v_p) == max(v_p) + min(v_p)
                        if current_sum[p] != current_max[p] + current_min[p]:
                            # If the condition fails for this prime, the subarray is not PE.
                            is_pe = False
                            # No need to check other primes for this subarray.
                            break 

                # If the current subarray is product equivalent, update the maximum length found so far.
                if is_pe:
                    max_len = max(max_len, current_len)

        # Return the length of the longest product equivalent subarray.
        return max_len