import math

class Solution:
  def minimumBeautifulSubstrings(self, s: str) -> int:
    n = len(s)

    # Precompute binary representations of powers of 5.
    # These are the only numbers whose binary representations can be "beautiful substrings".
    # A substring is beautiful if:
    # 1. It doesn't contain leading zeros (i.e., starts with '1', as 0 is not a power of 5).
    # 2. It's the binary representation of a number that is a power of 5.
    
    # Given N_max=15 for s.length:
    # 5^0 = 1 (binary "1")
    # 5^1 = 5 (binary "101")
    # 5^2 = 25 (binary "11001")
    # 5^3 = 125 (binary "1111101")
    # 5^4 = 625 (binary "1001110001")
    # 5^5 = 3125 (binary "110000110101")
    # 5^6 = 15625 (binary "11110100001001", length 14)
    # 5^7 = 78125 (binary "10011000100101101", length 17, which is > 15, so not relevant)
    
    # Hardcoding these is robust and clear for the given constraints.
    # Filter them by length to only include those not longer than s itself.
    all_powers_of_5_binaries = [
        "1", "101", "11001", "1111101", 
        "1001110001", "110000110101", "11110100001001"
    ]
    
    beautiful_strings_set = set()
    for b_str in all_powers_of_5_binaries:
        if len(b_str) <= n: # Only consider those whose length is not more than s's length
            beautiful_strings_set.add(b_str)

    # Memoization cache for the recursive function.
    # memo[k] will store the minimum number of beautiful substrings
    # to partition the suffix s[k:].
    memo = {}

    def solve(start_index: int) -> int:
        # Base case: If we have successfully partitioned the entire string.
        # The suffix starting at n is an empty string, which needs 0 substrings to form.
        if start_index == n:
            return 0
        
        # If the current character is '0', no beautiful substring can start at this index.
        # A beautiful string must start with '1' (as integer 0 is not a power of 5).
        if s[start_index] == '0':
            return math.inf # Using math.inf to represent impossibility
        
        # If we've already computed the result for this suffix, return it from cache.
        if start_index in memo:
            return memo[start_index]
        
        min_partitions_for_this_suffix = math.inf
        
        # Try all possible end points for a substring starting at start_index.
        # The substring is s[start_index ... end_index_inclusive].
        for end_index_inclusive in range(start_index, n):
            # Python slicing is s[start_index : end_index_exclusive]
            current_substring = s[start_index : end_index_inclusive + 1]
            
            # Check if this substring is a binary representation of a power of 5.
            # (It's guaranteed to not have leading zeros due to the s[start_index] == '0' check above)
            if current_substring in beautiful_strings_set:
                # If current_substring is beautiful, recursively solve for the rest of the string.
                # The rest of the string starts at (end_index_inclusive + 1).
                partitions_for_remaining_string = solve(end_index_inclusive + 1)
                
                # If the rest of the string can also be partitioned (i.e., result is not math.inf)
                if partitions_for_remaining_string != math.inf:
                    min_partitions_for_this_suffix = min(min_partitions_for_this_suffix, 1 + partitions_for_remaining_string)
        
        # Cache the result for this suffix.
        memo[start_index] = min_partitions_for_this_suffix
        return min_partitions_for_this_suffix

    # Start the recursive process for the whole string s (i.e., suffix s[0:]).
    min_total_partitions = solve(0)
    
    # If min_total_partitions is still math.inf, it means no such partition is possible.
    # Otherwise, it holds the minimum number of beautiful substrings.
    return min_total_partitions if min_total_partitions != math.inf else -1