from typing import List

# Definition of BIT class (Binary Indexed Tree / Fenwick Tree)
class BIT:
    """Binary Indexed Tree (Fenwick Tree) supporting point updates and range sum queries."""
    
    def __init__(self, size):
        """Initializes BIT with given size. All elements default to 0."""
        # The tree array uses 1-based indexing internally. Its size needs to be size+1.
        # We store N elements, indexed 0 to N-1. The tree represents these N elements.
        self.tree = [0] * (size + 1)
        # Store N, the number of elements the BIT operates on.
        self.size = size 

    def update(self, i, delta):
        """Adds delta to element at index i (0-based)."""
        # Convert 0-based index i to 1-based index for BIT operations
        i += 1 
        # Traverse up the tree structure by adding the least significant bit (LSB)
        # Update all nodes that cover index i.
        # Stop when index exceeds the tree array bounds (size N means indices 1..N).
        while i <= self.size: 
            self.tree[i] += delta
            # Move to the next index based on BIT structure (add LSB)
            i += i & (-i) 

    def query(self, i):
        """Queries the prefix sum up to index i (inclusive, 0-based)."""
        # Convert 0-based index i to 1-based index for BIT operations
        i += 1
        s = 0
        # Traverse down the tree structure by subtracting the LSB
        # Accumulate values from nodes contributing to the prefix sum up to i.
        # Stop when index becomes 0.
        while i > 0:
            s += self.tree[i]
            # Move to the parent/previous index based on BIT structure (subtract LSB)
            i -= i & (-i) 
        return s

    def query_range(self, l, r):
        """Queries the sum of elements in the range [l, r] (inclusive, 0-based)."""
        # Basic check for invalid range: if l > r, the range is empty, sum is 0.
        if l > r:
            return 0
        # The sum of range [l, r] is computed using prefix sums: PrefixSum(r) - PrefixSum(l-1).
        # The query(k) method calculates PrefixSum(k), sum of elements from index 0 to k.
        return self.query(r) - self.query(l - 1)

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        """
        Processes queries to count peaks in subarrays or update elements using a BIT.

        Args:
          nums: The initial integer array.
          queries: A list of queries. Each query is a list:
                   [1, l, r] for counting peaks in nums[l..r].
                   [2, index, val] for updating nums[index] to val.

        Returns:
          A list containing the results of all type 1 queries in the order they appear.

        Constraints and Definitions:
        - A peak at index i satisfies 0 < i < N-1 and nums[i] > nums[i-1] and nums[i] > nums[i+1].
        - For type 1 query [1, l, r], count peaks at index i such that l < i < r.
        - The first and last elements of an array or subarray cannot be peaks.
        - 3 <= nums.length <= 10^5
        - 1 <= nums[i] <= 10^5
        - 1 <= queries.length <= 10^5
        """
        N = len(nums)
        # Initialize BIT of size N. It will store peak indicators (0 or 1) at relevant indices.
        # The indices relevant for peaks are 1 to N-2. BIT indices correspond to nums indices.
        bit = BIT(N) 

        # Helper function to determine if an index k is a peak based on current nums array state.
        # It's defined inside the main method to easily capture N from the outer scope.
        def is_peak_at(k, current_nums):
            """Checks if index k is a peak in current_nums array."""
            # A peak must be strictly within the array bounds (index > 0 and index < N-1).
            if not (0 < k < N - 1):
                return 0
            # Peak condition: element must be strictly greater than both its neighbors.
            if current_nums[k] > current_nums[k-1] and current_nums[k] > current_nums[k+1]:
                return 1 # It's a peak
            else:
                return 0 # Not a peak

        # Initialize the BIT based on the initial state of the nums array.
        # Iterate through all possible indices where a peak could exist (1 to N-2).
        for k in range(1, N - 1):
            if is_peak_at(k, nums):
                # If k is initially a peak, mark it in the BIT by adding 1 at index k.
                bit.update(k, 1)
        
        # List to store the results of type 1 queries.
        results = []
        
        # Process each query sequentially.
        for query in queries:
            query_type = query[0]
            
            if query_type == 1:
                # Type 1 query: Calculate the number of peaks in the subarray nums[l..r].
                l, r = query[1], query[2]
                
                # According to problem statement & definition, a peak index i within subarray nums[l..r]
                # must satisfy l < i < r. This means we need to sum peak indicators
                # over the index range [l+1, r-1].
                
                # Use the BIT's range query capability for the required range.
                # The query_range method handles empty ranges (e.g., when l+1 > r-1) correctly, returning 0.
                count = bit.query_range(l + 1, r - 1)
                results.append(count)
            else: # query_type == 2
                # Type 2 query: Update the value at nums[index] to val.
                index, val = query[1], query[2]

                # Optimization: If the value at index doesn't actually change, there's nothing to update.
                if nums[index] == val:
                    continue

                # Identify indices whose peak status might change due to the update at 'index'.
                # An update at `index` can affect the peak status check for `index-1`, `index`, and `index+1`.
                # We only care about indices k such that 0 < k < N-1, as only these can be peaks.
                affected_indices = []
                # The loop range `max(1, index - 1)` to `min(N - 1, index + 2)` ensures we only consider valid potential peak indices k (1 <= k <= N-2)
                # that are neighbours of `index` or `index` itself.
                for k in range(max(1, index - 1), min(N - 1, index + 2)):
                     affected_indices.append(k)

                # Store the peak status of these potentially affected indices *before* the update.
                old_statuses = {}
                for k in affected_indices:
                    old_statuses[k] = is_peak_at(k, nums)

                # Perform the actual update on the nums array. This changes the state used by is_peak_at.
                nums[index] = val

                # Re-calculate peak statuses for the affected indices *after* the update.
                # Compare with old statuses and update the BIT accordingly.
                for k in affected_indices:
                    new_status = is_peak_at(k, nums)
                    # Retrieve the old status for comparison. k must be in old_statuses dictionary.
                    old_status = old_statuses[k] 
                    
                    # If the peak status changed for index k...
                    if new_status != old_status:
                        # Calculate the change: +1 if it became a peak, -1 if it ceased to be one.
                        delta = new_status - old_status 
                        # Apply this change (delta) to the BIT at index k.
                        bit.update(k, delta)
        
        # Return the collected results of all type 1 queries in the order they were processed.
        return results