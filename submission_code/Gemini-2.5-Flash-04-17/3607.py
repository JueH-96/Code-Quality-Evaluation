from typing import List

# Precompute smallest prime factors using Sieve of Eratosthenes
MAX_VAL = 10**6
spf = [i for i in range(MAX_VAL + 1)]

# spf[0] and spf[1] are not used in the logic for numbers >= 1.
# For numbers > 1:
# We only need to sieve up to sqrt(MAX_VAL) because if a number n has a prime factor
# greater than sqrt(n), it can have at most one such prime factor (n = p1 * p2 * ...).
# If all prime factors were > sqrt(n), then n would be > (sqrt(n))^k for k>=2, which implies n > n.
# So, the smallest prime factor of n must be <= sqrt(n) unless n itself is prime.
# Our sieve initializes spf[i] = i, so primes will retain spf[i] = i.
# Composite numbers j will have spf[j] updated by their smallest prime factor i <= sqrt(j).
for i in range(2, int(MAX_VAL**0.5) + 1):
    if spf[i] == i: # i is prime
        # Mark multiples of i starting from i*i
        # Use 1LL*i*i to prevent overflow if MAX_VAL is large, although 10^6*10^6 is too big.
        # Here MAX_VAL = 10^6, so i*i can be up to (10^3)^2 = 10^6. Fits in int.
        for j in range(i * i, MAX_VAL + 1, i):
            if spf[j] == j: # If spf[j] is not set yet, set it to i
                spf[j] = i

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        total_operations = 0

        # We process the array from right to left.
        # The last element nums[n-1] has no constraint from the right.
        # To minimize operations for nums[n-1], we should pick the value reachable with 0 operations,
        # which is nums[n-1] itself.
        # So, the chosen value for the last element is nums[n-1].
        prev_v = nums[n-1]

        # Iterate from the second to last element backwards
        for i in range(n - 2, -1, -1):
            current_num = nums[i]
            
            # We want to find the first value v in the sequence [current_num, spf(current_num), spf(spf(current_num)), ...]
            # such that v <= prev_v. The number of operations to reach v is its position in this sequence (0-indexed).
            # The sequence of reachable values is:
            # value_0 = current_num (0 operations)
            # value_1 = spf(value_0) (1 operation, if value_0 > 1)
            # value_2 = spf(value_1) (2 operations, if value_1 > 1)
            # ...
            # This sequence continues until the value becomes 1 or prime. Once it's 1 or prime, spf(value) = value
            # (for prime) or spf is undefined (for 1), and the operation x -> x / (x / spf(x)) = spf(x) doesn't yield
            # a smaller value > 1. The sequence of *distinct* reachable values stops at the first 1 or prime.

            temp_val = current_num # Start check with the original value (0 ops)
            ops_needed = 0
            found_suitable_value = False

            while True:
                # Check if the current value in the sequence is suitable (<= prev_v)
                if temp_val <= prev_v:
                    found_suitable_value = True
                    break

                # If the current value (temp_val) is > prev_v, we must try to reduce it.
                # We can only reduce temp_val if it's > 1 and composite.
                # If temp_val is 1 or prime, it cannot be reduced to a *smaller* value > 1.
                # If temp_val > prev_v and cannot be reduced further (1 or prime > 1),
                # then no reachable value will be <= prev_v.
                
                # temp_val == 1: Cannot reduce 1 further. If 1 > prev_v, impossible.
                # spf[temp_val] == temp_val: temp_val is prime. Operation results in temp_val itself.
                # In both cases (1 or prime), we cannot get a strictly smaller positive integer.
                # If temp_val is currently > prev_v, and it's 1 or prime, we are stuck.
                if temp_val == 1 or spf[temp_val] == temp_val:
                     break # No suitable value found in the sequence [current_num, spf(current_num), ...]

                # If temp_val > prev_v and temp_val is composite > 1, apply the operation
                # The operation x -> spf(x) gives the next value in the sequence.
                temp_val = spf[temp_val]
                ops_needed += 1

            if not found_suitable_value:
                # If the loop finished without finding a suitable value, it's impossible.
                # This happens if the smallest reachable value (1 or prime) is still > prev_v.
                return -1

            # Found the first suitable value (temp_val) which required ops_needed operations
            total_operations += ops_needed
            prev_v = temp_val # This is the chosen value for nums[i] for the next iteration

        return total_operations