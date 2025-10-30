import sys
from typing import List

# Attempt to set a higher recursion depth limit. This is necessary because the
# path compression in the Disjoint Set Union (DSU) find operation is implemented
# recursively. In the worst case, the recursion depth could be proportional to
# the number of elements or the difference between height values, potentially
# exceeding Python's default limit (often around 1000). N = 10^5 is the maximum
# number of towers, which gives a reasonable upper bound for the practical depth needed.
try:
    # Set recursion depth limit slightly higher than N (max number of towers = 10^5)
    sys.setrecursionlimit(10**5 + 10) 
except Exception as e:
    # If setting the recursion limit fails (e.g., due to environment restrictions),
    # the code might still work for smaller inputs or inputs that don't create deep
    # recursion paths. However, for worst-case inputs, it could lead to a RecursionError.
    # An iterative implementation of the 'find' function would be a more robust alternative
    # in environments where recursion depth is strictly limited.
    # print(f"Warning: Could not set recursion depth limit. {e}")
    pass 

class Solution:
    """
    Solves the problem of assigning distinct positive integer heights to towers
    such that the height of the i-th tower is at most maximumHeight[i],
    and the total sum of heights is maximized. Uses a greedy approach combined
    with Disjoint Set Union (DSU) optimized with path compression.
    """
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        """
        Calculates the maximum possible total sum of tower heights according to the rules.

        Args:
            maximumHeight: A list where maximumHeight[i] is the maximum allowed height for tower i.

        Returns:
            The maximum possible total sum of assigned tower heights, or -1 if it's impossible
            to assign heights satisfying all conditions.
        """
        n = len(maximumHeight)
        
        # Create pairs of (max_h, original_index) to process towers based on their height limits.
        # Storing the original index isn't strictly necessary for calculating the sum,
        # but it's good practice if we needed to return the actual assignment.
        pairs = []
        for i in range(n):
            pairs.append((maximumHeight[i], i))
            
        # Sort the pairs in descending order based on maximumHeight.
        # The greedy strategy is to prioritize towers that can potentially have larger heights.
        # By assigning heights to these towers first, we try to use larger height values overall.
        pairs.sort(key=lambda x: x[0], reverse=True)
        
        # The 'parent' dictionary implements the DSU structure.
        # It maps a height 'h' to the next available height strictly less than 'h'.
        # If 'h' is available, it maps to itself (parent[h] = h) or is not yet in the dictionary.
        # This allows efficiently finding the largest available height less than or equal to a target value.
        parent = {} 
        
        def find(h):
            """
            Performs the DSU find operation with path compression.
            It returns the largest available height less than or equal to 'h'.
            If no positive height is available, it returns 0.
            """
            if h <= 0:
                # Base case: If we search for heights <= 0, there are no positive available heights.
                return 0 
            
            # If 'h' is not in the parent dictionary, it means 'h' has not been used or accessed yet.
            # Therefore, 'h' is available. We mark it as available by setting parent[h] = h
            # and return 'h'.
            if h not in parent:
                parent[h] = h
                return h
                
            # If parent[h] == h, it means 'h' is currently marked as available.
            if parent[h] == h:
                return h
                
            # If parent[h] < h, it means 'h' has been used previously.
            # The value parent[h] stores a link towards the next available height below 'h'.
            # This link was established when 'h' was marked as used (pointing to find(h-1)).
            # We recursively call find on parent[h] to follow the chain and find the actual available height.
            res = find(parent[h])
            
            # Path Compression: Update parent[h] to point directly to the final result 'res'.
            # This flattens the structure for future searches involving 'h', making them faster.
            parent[h] = res 
            return res

        total_sum = 0
        
        # Iterate through the towers, processed in order of decreasing maximum height limit.
        for max_h, _ in pairs: # The original index is not needed for the sum calculation itself.
            
            # Find the largest available height 'assigned_h' that is less than or equal to 'max_h'.
            assigned_h = find(max_h)
            
            # If assigned_h is 0, it signifies that no positive integer height <= max_h is available.
            # Since towers are processed greedily based on max height, if the current tower cannot
            # be assigned a valid height, then no overall valid assignment is possible.
            if assigned_h == 0:
                return -1
                
            # Add the successfully found available height to the total sum.
            total_sum += assigned_h
            
            # Mark the height 'assigned_h' as used in our DSU structure.
            # This is achieved by setting parent[assigned_h] to point to the result of find(assigned_h - 1).
            # This effectively connects the used height 'assigned_h' to the next available slot below it.
            # Any future 'find' operation involving 'assigned_h' will now be redirected to search below it.
            parent[assigned_h] = find(assigned_h - 1)
            
        # If all towers have been successfully assigned a distinct positive height, return the maximized total sum.
        return total_sum