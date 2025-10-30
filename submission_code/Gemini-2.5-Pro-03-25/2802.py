import sys
# Potentially increase recursion depth limit if needed, although unlikely for this problem constraints.
# sys.setrecursionlimit(2000) 

# Precomputation placed outside the class definition.
# This code will run once when the Python module is loaded.
_MAX_N = 1000
_punishment_cache = {}  # Cache results of _check_partition_static(i) -> bool
# _precomputed_sums[k] will store the Punishment Number for input n=k.
_precomputed_sums = [0] * (_MAX_N + 1)  # Stores Punishment Number Sum up to index i

def _check_partition_static(i: int) -> bool:
    """
    Helper function (static method simulation) to check if integer 'i' satisfies the partition condition.
    An integer 'i' satisfies the condition if the decimal representation of its square (i*i) 
    can be partitioned into contiguous substrings such that the sum of the integer values 
    of these substrings equals 'i'.
    Uses memoization for the check itself and within the Depth First Search (DFS).
    """
    # Check cache for the final boolean result for 'i'. If already computed, return it.
    # This cache stores the final True/False result for whether 'i' satisfies the condition.
    if i in _punishment_cache:
        return _punishment_cache[i]

    s = str(i * i)  # The string representation of the square of i.
    target = i      # The target sum we need to achieve by partitioning 's'.
    n_len = len(s)  # Length of the string representation.
    
    # Memoization dictionary for DFS states (index, current_target).
    # This memoization is specific to the current call for 'i'.
    memo = {}  

    def dfs(index: int, current_target: int) -> bool:
        """
        Depth First Search function to determine if the suffix of the string s[index:] 
        can be partitioned into substrings that sum up to 'current_target'.
        """
        # Check memoization table for the current state (index, current_target).
        state = (index, current_target)
        if state in memo:
            return memo[state]

        # Base case: If current_target becomes negative, this path is invalid as we overshot the sum.
        if current_target < 0:
            return False
        
        # Base case: If we have successfully processed the entire string (index reaches the end).
        if index == n_len:
            # Return True if the target sum is exactly met (current_target is 0), False otherwise.
            return current_target == 0

        # Recursive step: Try partitioning the string starting from 'index'.
        res = False  # Flag to indicate if a valid partition is found from this state.
        num_val = 0  # Represents the integer value of the current substring s[index...j].
        
        # Iterate through all possible end points 'j' for the current partition segment.
        # The segment being considered is s[index...j].
        for j in range(index, n_len):
            # Build the integer value of the substring s[index...j] incrementally.
            num_val = num_val * 10 + int(s[j])
            
            # Optimization: Pruning. If the current number segment's value exceeds 
            # the remaining target sum, any further extension of this segment (making the number larger) 
            # will also exceed the target. Since partition values are positive, we can stop exploring 
            # further extensions from 'j' in this iteration.
            if num_val > current_target:
                break
            
            # Recursively call dfs for the remaining part of the string (starting from j+1)
            # with the target sum reduced by the value of the current segment (num_val).
            if dfs(j + 1, current_target - num_val):
                # If the recursive call returns True, it means a valid partition is found down this path.
                res = True
                # We can break the loop as we only need to find if at least one valid partition exists.
                break 

        # Store the result for the current state (index, current_target) in the memoization table.
        memo[state] = res
        return res

    # Initial call to DFS starts from the beginning of the string (index 0) 
    # with the original target sum 'i'.
    result = dfs(0, target)
    
    # Cache the final result (True/False) for 'i' in the global cache.
    _punishment_cache[i] = result
    return result

# --- Precomputation Execution ---
# This loop runs once when the script/module is loaded. It calculates the 
# punishment numbers for all n from 1 to _MAX_N.
_current_punishment_sum = 0
for i in range(1, _MAX_N + 1):
    # Check if the integer 'i' satisfies the punishment number condition.
    if _check_partition_static(i):
        # If it satisfies, add its square to the cumulative punishment sum.
        _current_punishment_sum += i * i
    # Store the cumulative punishment sum up to 'i'. 
    # _precomputed_sums[k] will hold the Punishment Number for n=k.
    _precomputed_sums[i] = _current_punishment_sum


# Class definition as required by the problem format.
class Solution:
    def punishmentNumber(self, n: int) -> int:
        """
        Calculates the punishment number of n.
        The punishment number is the sum of the squares of all integers i (1 <= i <= n)
        such that the decimal representation of i * i can be partitioned into contiguous 
        substrings whose integer values sum up to i.
        
        This implementation relies on precomputed results stored in the global list _precomputed_sums,
        making the method call itself O(1) time complexity after the initial setup phase.
        """
        # Constraints state 1 <= n <= 1000.
        # The result for the given n is directly retrieved from the precomputed list.
        return _precomputed_sums[n]