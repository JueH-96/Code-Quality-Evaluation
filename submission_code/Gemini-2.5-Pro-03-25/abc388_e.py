# YOUR CODE HERE
import sys

# Attempt to import SortedList from sortedcontainers package.
# The efficient O(N log N) solution relies on this library.
# If it's not available on the execution environment, the provided fallback 
# class (`SortedList` using `list` and `bisect`) will be used instead.
# However, the fallback class has O(N) complexity for the `remove` operation,
# leading to an overall O(N^2) time complexity, which is likely too slow for the given constraints.
try:
    # Attempt import from the optimized library
    from sortedcontainers import SortedList
except ImportError:
    # Fallback implementation if sortedcontainers is not installed.
    # WARNING: This implementation has suboptimal performance characteristics.
    import bisect
    
    class SortedList:
        """
        A minimalistic SortedList implementation using Python's built-in list and bisect module.
        Provides basic sorted list functionality but with O(N) complexity for element removal.
        """
        def __init__(self, iterable=None):
            self._list = []
            if iterable:
                # Initialize with elements from iterable and sort them.
                self._list.extend(iterable)
                self._list.sort()

        def add(self, value):
            # Insert value while maintaining sort order using binary search based insertion. O(N) due to list insertion.
            bisect.insort_left(self._list, value)

        def remove(self, value):
            # Find the index of the first occurrence of value using binary search. O(log N)
            idx = bisect.bisect_left(self._list, value)
            # Check if the value is actually present at the found index.
            if idx < len(self._list) and self._list[idx] == value:
                # Remove the element by index. list.pop(idx) takes O(N) time.
                self._list.pop(idx) 
            else:
                 # If value not found, raise ValueError, mimicking behavior of sortedcontainers.
                 raise ValueError(f"{value} not found in list")

        def bisect_right(self, value):
            # Find the index where value should be inserted to maintain order,
            # returning the index of the first element strictly greater than value. O(log N)
            return bisect.bisect_right(self._list, value)

        def __len__(self):
            # Return the number of elements in the list. O(1)
            return len(self._list)

        def __getitem__(self, index):
            # Access element by index. O(1)
            return self._list[index]
        
        def __contains__(self, value):
             # Check if value is present in the list using binary search. O(log N)
             idx = bisect.bisect_left(self._list, value)
             return idx < len(self._list) and self._list[idx] == value

# Function to solve the kagamimochi problem
def solve():
    # Read the number of mochi, N, from standard input.
    N = int(sys.stdin.readline())
    
    # Handle edge case where N < 2. According to constraints N >= 2, but good practice.
    if N < 2:
        print(0)
        return

    # Read the sizes of the N mochi, A, from standard input.
    # The input guarantees A is sorted in ascending order.
    A = list(map(int, sys.stdin.readline().split()))

    # Create a SortedList to keep track of available mochi.
    # Store elements as pairs (value, index), where 'value' is the mochi size A[k]
    # and 'index' is the original 0-based index k.
    # The SortedList automatically maintains these pairs sorted primarily by value,
    # and secondarily by index for tie-breaking.
    available = SortedList([(A[k], k) for k in range(N)])
    
    # Initialize the count of kagamimochi pairs formed.
    K = 0
    
    # Iterate through potential bottom mochi indices `j` from largest (N-1) down to smallest (0).
    # This implements the greedy strategy: "pair the largest available bottom mochi
    # with the largest available top mochi that satisfies the condition".
    for j in range(N - 1, -1, -1):
        
        current_val_j = A[j] # Size of the current potential bottom mochi.
        
        # Check if the mochi `j` (represented by pair (A[j], j)) is currently in the `available` list.
        # If it's not present, it means this mochi was already used as a top piece `i`
        # for some previously processed larger index `k > j`. In this case, skip `j`.
        # The `__contains__` check on SortedList takes O(log N) time.
        if (current_val_j, j) not in available:
             continue 

        # Mochi `j` is available. Remove it from the `available` list because we are now considering it.
        # Regardless of whether it forms a pair or not, it will not be available for future consideration.
        # The `remove` operation on SortedList takes O(log N) time (for the optimized library version).
        available.remove((current_val_j, j))
        
        # Calculate the maximum allowed size for a mochi `i` to be placed on top of mochi `j`.
        # The condition is A[i] <= A[j] / 2. Integer division `//` is appropriate here.
        target_val = current_val_j // 2
        
        # Search within the `available` list for a suitable top mochi `i`.
        # We need the mochi `i` with the largest index such that its size `A[i] <= target_val`.
        # The `bisect_right` method helps find candidates efficiently. We search for `(target_val, N)`.
        # `N` is used as a proxy for infinity for the index part of the pair, ensuring that
        # the binary search correctly finds the upper bound based on `target_val`.
        # All elements at indices less than `idx` will have `value <= target_val`.
        # `bisect_right` on SortedList takes O(log N) time.
        idx = available.bisect_right((target_val, N)) 
        
        # If `idx > 0`, it means there is at least one available mochi `i` satisfying `A[i] <= target_val`.
        if idx > 0:
            # The element at index `idx - 1` in the SortedList is the best candidate according to our greedy strategy.
            # It has the largest value component `val_i <= target_val`. Among elements with the same maximum `val_i`,
            # it has the largest index `i` due to the secondary sort key.
            # Accessing element by index `__getitem__` on SortedList takes O(log N) time.
            val_i, i = available[idx - 1]
            
            # A pair (i, j) can be formed. Increment the kagamimochi count K.
            K += 1
            
            # Remove the used top mochi `i` (represented by pair (val_i, i)) from the `available` list.
            # The `remove` operation takes O(log N) time.
            available.remove((val_i, i))
            
        # Else (idx == 0): No suitable top mochi `i` was found among the available ones for the current bottom mochi `j`.
        # Mochi `j` has already been removed from `available`, so it's effectively discarded. No further action needed for `j`.
            
    # After iterating through all potential bottom mochi, print the final maximum count K.
    print(K)

# Execute the solve function to start the process.
solve()