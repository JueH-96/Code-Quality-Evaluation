from typing import List

class Solution:
    """
    Solves the problem of finding the earliest second to mark all indices using binary search
    and a greedy check function.
    """
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        """
        Finds the earliest second `k` such that all indices in `nums` can be marked
        using operations up to second `k`.

        Args:
          nums: A list of integers representing initial values at indices. 
                The problem uses 1-based indexing for indices, but this list is 0-based.
                Length n corresponds to indices 1 to n.
          changeIndices: A list of integers representing the index (1-based) targeted at each second.
                         Length m corresponds to seconds 1 to m.

        Returns:
          The earliest second (1-based) if possible, otherwise -1.
        """
        n = len(nums)
        m = len(changeIndices)

        # check(k) function determines if it is possible to mark all indices using the first k seconds (1 to k).
        def check(k: int) -> bool:
            """
            Checks if it's possible to mark all indices using the first k seconds.
            This function implements a greedy strategy based on marking each index at its latest possible occurrence.

            Args:
              k: The number of seconds available (seconds 1 to k).

            Returns:
              True if possible within k seconds, False otherwise.
            """
            
            # `last[j]` stores the latest second (0-based index, s from 0 to k-1, corresponding to seconds 1 to k) 
            # where index j (0-based) appears in `changeIndices[0...k-1]`.
            last = [-1] * n 
            for s in range(k):
                # `changeIndices` contains 1-based indices. Convert to 0-based index `idx`.
                idx = changeIndices[s] - 1 
                last[idx] = s # Update last seen second for index idx

            # Check if all indices 0..n-1 have at least one occurrence within the first k seconds.
            # If an index `j` never appears in `changeIndices[0...k-1]`, it's impossible to mark it.
            for j in range(n):
                if last[j] == -1:
                    return False

            # Calculate the total number of decrement operations needed across all indices.
            # Python's arbitrary precision integers handle potentially large sums.
            total_decrements = 0
            for x in nums:
                total_decrements += x

            # Check a necessary condition: the total number of operations required must be <= k.
            # Total operations = sum of decrements needed for all indices + n marking operations.
            # Each decrement and each mark operation requires one second.
            if total_decrements + n > k:
                 # Not enough seconds available in total, even ignoring timing constraints.
                 return False

            # Simulation to check feasibility considering timing constraints.
            # The greedy strategy is to mark each index `j` at second `last[j]`.
            # `available_slots` tracks the number of seconds encountered so far that are *not yet*
            # committed/reserved for fulfilling the requirements (decrements + mark) of some index.
            available_slots = 0
            
            for s in range(k): # Iterate through seconds s=0 to k-1 (corresponds to time 1 to k)
                # Each second provides one potential slot that can be used for an operation (decrement or mark).
                available_slots += 1 
                
                # Determine the index (0-based) targeted by the operation at second s+1.
                idx_target = changeIndices[s] - 1 
                
                # Check if this second `s` is the designated marking time for `idx_target`.
                # The designated marking time is the latest occurrence `last[idx_target]`.
                if s == last[idx_target]: 
                    
                    # `cost` is the number of decrements needed for index `idx_target`.
                    cost = nums[idx_target] 
                    
                    # Total slots needed by time `s` for index `idx_target`: `cost` for decrements + 1 for the mark operation.
                    # These operations must logically complete by second `s`.
                    # We check if the accumulated `available_slots` up to second `s` is sufficient.
                    if available_slots >= cost + 1:
                        # Sufficient slots available. Reserve `cost + 1` slots for index `idx_target`.
                        available_slots -= (cost + 1) 
                    else:
                        # Not enough available slots accumulated by time `s` to cover the costs.
                        # This strategy fails for this `k`.
                        return False
            
            # If the loop completes without returning False, it means a valid schedule exists
            # under the strategy of marking each index at its latest possible time within the first k seconds.
            return True

        # Binary search for the minimum k in the range [1, m].
        low = 1   # Minimum possible answer is 1 second.
        high = m  # Maximum possible answer is m seconds.
        ans = -1  # Initialize answer to -1 (impossible case).

        while low <= high:
            mid = (low + high) // 2 # Calculate midpoint for binary search.
            
            # Check if it's possible to mark all indices within `mid` seconds.
            if check(mid):
                # If possible, `mid` is a potential answer.
                # Try to find an even smaller `k` that works.
                ans = mid 
                high = mid - 1 # Search in the lower half [low, mid-1].
            else:
                # If not possible with `mid` seconds, we need more time.
                # Search in the upper half [mid+1, high].
                low = mid + 1 

        # Return the minimum `k` found, or -1 if no `k` worked.
        return ans