import math
from typing import List

# Using float('inf') for infinity representation
POS_INF = float('inf')

class Solution:
    """
    Finds the minimum possible maximum absolute difference between adjacent elements 
    after replacing -1s with a chosen pair of positive integers (x, y).
    
    The core idea is to use binary search on the possible maximum difference `k`.
    For a given `k`, we check if there exists a pair (x, y) such that we can replace
    each -1 with either x or y, and all adjacent absolute differences are at most `k`.
    """
    def minDifference(self, nums: List[int]) -> int:
        """
        Main method to calculate the minimum difference using binary search.
        """
        n = len(nums)

        # Edge case: If all elements are -1, we can choose x=y=1 (or any identical positive integer pair), 
        # making all elements the same. The differences will be 0. The minimum max difference is 0.
        if all(x == -1 for x in nums):
            return 0

        def check(k: int) -> bool:
            """
            Checks if it's possible to achieve a maximum adjacent difference of at most k.
            Returns True if possible, False otherwise.
            
            Args:
                k: The target maximum absolute difference.
            
            Returns:
                bool: True if k is achievable, False otherwise.
            """
            # Step 1: Check adjacent non-`-1` pairs & determine if |x-y|<=k is required.
            # If any adjacent non-`-1` pair has difference > k, then k is impossible.
            # If any adjacent pair is `(-1, -1)`, it implies a constraint `|x-y| <= k`.
            require_xy_diff_le_k = False
            for i in range(n - 1):
                if nums[i] != -1 and nums[i+1] != -1:
                    # If existing adjacent elements already violate the difference k
                    if abs(nums[i] - nums[i+1]) > k:
                        return False
                # Check if any adjacent pair is (-1, -1), this imposes the constraint |x-y| <= k.
                if nums[i] == -1 and nums[i+1] == -1:
                    require_xy_diff_le_k = True

            # Step 2: Compute all required intervals [L_i, R_i] for each nums[i] == -1.
            # An interval [L_i, R_i] defines the valid range for the value replacing nums[i],
            # based on constraints imposed by its non-`-1` neighbors.
            intervals = [] # Stores pairs [L_i, R_i] for each -1 element.
            indices_s = [] # Stores indices i where nums[i] == -1. This represents the set S.
            
            for i in range(n):
                if nums[i] == -1:
                    indices_s.append(i)
                    
                    # Initialize potential range for replacement value at index i.
                    # The replacement must be a positive integer.
                    current_L = 1
                    current_R = POS_INF # Use float('inf') for potentially unbounded upper range

                    # Apply constraints from neighbors if they exist and are not -1.
                    # The replacement value `z` must satisfy |z - neighbor_val| <= k.
                    if i > 0 and nums[i-1] != -1:
                        current_L = max(current_L, nums[i-1] - k)
                        current_R = min(current_R, nums[i-1] + k)
                    
                    if i < n - 1 and nums[i+1] != -1:
                        current_L = max(current_L, nums[i+1] - k)
                        current_R = min(current_R, nums[i+1] + k)
                    
                    # Ensure the lower bound is at least 1, as x and y must be positive integers.
                    current_L = max(1, current_L) 

                    # If at any point the range becomes invalid (L > R), then k is too small.
                    if current_L > current_R:
                         return False
                    intervals.append([current_L, current_R])
            
            # Step 3: If S is empty (no -1s), and step 1 passed, then k is achievable.
            if not indices_s: 
                 return True

            # Step 4: Compute the intersection of all intervals [L_i, R_i].
            # Let this intersection be [max_L, min_R].
            # max_L is the maximum of all lower bounds L_i.
            # min_R is the minimum of all upper bounds R_i.
            max_L = 1 # Initialize max_L to 1 (minimum possible value for any L_i)
            min_R = POS_INF
            for L, R in intervals:
                 max_L = max(max_L, L)
                 min_R = min(min_R, R)
            
            # If the intersection is non-empty (max_L <= min_R), we can choose x = y = max_L.
            # This satisfies all interval constraints (since max_L is in every [L_i, R_i])
            # and satisfies |x-y|=0 <= k. So k is achievable.
            if max_L <= min_R:
                 return True 

            # Step 5-7: The overall intersection is empty (max_L > min_R).
            # This means we cannot use a single value (x=y) to replace all -1s.
            # We must use two distinct values x and y.
            # We need to determine if there exist x and y such that:
            # 1. x, y >= 1
            # 2. For every interval [L_i, R_i], either x is in [L_i, R_i] or y is in [L_i, R_i].
            # 3. If require_xy_diff_le_k is True, then |x-y| <= k.
            #
            # We derived a condition check based on candidate values related to max_L and min_R.
            # Let I_x be the intersection of intervals not covered by max_L.
            # Let I_y be the intersection of intervals not covered by min_R.
            # A valid pair (x,y) must exist such that x is in I_x and y is in I_y.

            # Calculate I_y = [L_y, R_y], the intersection of intervals NOT containing min_R.
            # If min_R is contained in an interval, that interval imposes no constraint on y if we assume x=min_R.
            L_y = 1
            R_y = POS_INF
            for L, R in intervals:
                 # An interval [L, R] does not contain min_R if min_R < L or min_R > R.
                 if min_R < L or min_R > R: 
                      L_y = max(L_y, L)
                      R_y = min(R_y, R)
            
            # Calculate I_x = [L_x, R_x], the intersection of intervals NOT containing max_L.
            L_x = 1
            R_x = POS_INF
            for L, R in intervals:
                 # An interval [L, R] does not contain max_L if max_L < L or max_L > R.
                 if max_L < L or max_L > R: 
                      L_x = max(L_x, L)
                      R_x = min(R_x, R)

            # If either required intersection I_x or I_y is empty (invalid range),
            # then it's impossible to find such x and y. k is not achievable.
            if L_x > R_x or L_y > R_y:
                return False 

            # Step 8: Check if we can find x in I_x=[L_x, R_x] and y in I_y=[L_y, R_y] 
            # such that |x-y| <= k, but only if this constraint is required.
            if require_xy_diff_le_k:
                 # The condition for existence of such x, y is max(L_x - R_y, L_y - R_x) <= k.
                 # This checks if the intervals I_x and I_y are "close enough" (within distance k).
                 if max(L_x - R_y, L_y - R_x) > k:
                     return False

            # If we have passed all checks, it implies that a valid pair (x,y) exists for this k.
            return True

        # Binary search on the answer k (the minimum possible maximum difference).
        # The range for k is [0, 10^9].
        low = 0
        high = 10**9 
        ans = high # Initialize answer to the maximum possible value

        while low <= high:
            mid = (low + high) // 2
            if check(mid):
                # If k=mid works, it's a possible answer. Try for an even smaller k.
                ans = mid
                high = mid - 1
            else:
                # If k=mid doesn't work, we need to allow a larger difference.
                low = mid + 1
                
        return ans