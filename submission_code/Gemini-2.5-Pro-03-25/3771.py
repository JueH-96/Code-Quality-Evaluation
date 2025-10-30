import math

# Sparse Table implementation for Range Minimum/Maximum Query (RMQ)
# Provides O(1) query time after O(n log n) preprocessing.
class SparseTable:
    def __init__(self, arr, func):
        """
        Initializes the Sparse Table.
        arr: Input list of numbers.
        func: The function to apply (min or max).
        """
        self.func = func
        self.n = len(arr)
        
        # Handle empty array case gracefully, though constraints state n >= 2
        if self.n == 0:
            self.log_n = -1
            self.st = []
            self.log2_lookup = []
            return

        # Calculate floor(log2(n)) for table dimensions
        # self.n.bit_length() - 1 is equivalent to floor(log2(self.n)) for n > 0
        self.log_n = self.n.bit_length() - 1 
        
        # Initialize sparse table structure: st[k][i] stores func(arr[i...i + 2^k - 1])
        self.st = [[0] * self.n for _ in range(self.log_n + 1)]
        
        # Base case: k=0 (intervals of length 2^0 = 1)
        for i in range(self.n):
            self.st[0][i] = arr[i]
            
        # Build the table for k > 0 using dynamic programming
        # For each level k, compute results based on level k-1
        for k in range(1, self.log_n + 1):
            # Interval length is 2^k
            # Ensure the right endpoint doesn't exceed array bounds: i + 2^k <= n
            for i in range(self.n - (1 << k) + 1):
                # Combine results from two overlapping intervals of size 2^(k-1)
                left_val = self.st[k-1][i]
                # The right interval starts at i + 2^(k-1)
                right_val = self.st[k-1][i + (1 << (k-1))]
                self.st[k][i] = self.func(left_val, right_val)

        # Precompute lookup table for floor(log2(x)) for O(1) query time
        # Size n+1 to handle lengths from 1 to n
        self.log2_lookup = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            # Efficiently compute log2 using previous value: log2(i) = log2(i/2) + 1
            self.log2_lookup[i] = self.log2_lookup[i // 2] + 1

    def query(self, l, r):
        """
        Queries the range [l, r] inclusive using the precomputed table.
        Returns func(arr[l...r]). Assumes 0 <= l <= r < n.
        """
        # Handle empty range query (e.g., if binary search leads to l > r)
        if l > r:
            # Return the identity element for the function (inf for min, -inf for max)
            return float('inf') if self.func == min else float('-inf')
        
        # Calculate the length of the query range
        length = r - l + 1
        # Get k = floor(log2(length)) using the precomputed lookup table
        k = self.log2_lookup[length]
        
        # Combine results from two overlapping intervals of size 2^k
        # The intervals are [l, l + 2^k - 1] and [r - 2^k + 1, r]
        # This covers the entire range [l, r]
        left_val = self.st[k][l]
        right_val = self.st[k][r - (1 << k) + 1]
        return self.func(left_val, right_val)


class Solution:
    """
    Solves the problem of finding if k disjoint special substrings exist in a string s.
    A substring s[i:j+1] is "special" if:
    1. Any character present inside the substring does not appear outside it in the string s.
    2. The substring is not the entire string s.
    """
    def maxSubstringLength(self, s: str, k: int) -> bool:
        """
        Determines if it's possible to select k disjoint special substrings.
        s: The input string.
        k: The required number of disjoint special substrings.
        Returns: True if possible, False otherwise.
        """
        
        n = len(s)
        
        # --- Handle Edge Cases ---
        # If k is 0, it's always possible (select zero substrings).
        if k == 0:
            return True
            
        # Constraints state n >= 2. If n < 2, no substring can be shorter than s,
        # so no special substrings are possible if k > 0.
        if n < 2:
             return False 

        # --- Step 1: Precompute first and last occurrences of each character ---
        # These define the minimum range containing all occurrences of a character.
        first = {}
        last = {}
        for i, char in enumerate(s):
            if char not in first:
                first[char] = i
            last[char] = i

        # Create arrays of first/last occurrences for each position in s.
        # These arrays will be used as input for the Sparse Tables.
        first_val = [first[char] for char in s]
        last_val = [last[char] for char in s]

        # --- Step 2: Build Sparse Tables for Range Min/Max Queries ---
        # Allows efficient querying of min(first) and max(last) within any range [i, j].
        # Preprocessing time: O(n log n). Query time: O(1).
        min_st = SparseTable(first_val, min) # For finding min first occurrence
        max_st = SparseTable(last_val, max)  # For finding max last occurrence

        # --- Step 3: Calculate i_start[j] for all j ---
        # i_start[j] = smallest index i such that the interval [i, j] contains 
        # the full extent of all characters within s[i...j] based on their last occurrence.
        # Formally: smallest i <= j such that max(last[s[p]] for p in [i..j]) <= j.
        # We find this using binary search on 'i' for each 'j'. Total time O(n log n).
        i_start = [-1] * n 
        for j in range(n):
            low = 0
            high = j
            # Initialize ans to an invalid start index (j+1)
            ans = j + 1 
            while low <= high:
                mid = (low + high) // 2
                # Query max(last) in range [mid, j] using Sparse Table O(1)
                max_last_in_range = max_st.query(mid, j)
                
                # If max_last is within [mid, j] (<= j), 'mid' is a possible start. Try smaller i.
                if max_last_in_range <= j:
                    ans = mid
                    high = mid - 1
                else:
                    # If max_last > j, 'mid' starts too early. Need to start later.
                    low = mid + 1
            i_start[j] = ans

        # --- Step 4: Calculate j_end[i] for all i ---
        # j_end[i] = largest index j such that the interval [i, j] contains 
        # the full extent of all characters within s[i...j] based on their first occurrence.
        # Formally: largest j >= i such that min(first[s[p]] for p in [i..j]) >= i.
        # We find this using binary search on 'j' for each 'i'. Total time O(n log n).
        j_end = [-1] * n 
        for i in range(n):
            low = i
            high = n - 1
            # Initialize ans to an invalid end index (i-1)
            ans = i - 1
            while low <= high:
                mid = (low + high) // 2
                # Query min(first) in range [i, mid] using Sparse Table O(1)
                min_first_in_range = min_st.query(i, mid)

                # If min_first is within [i, mid] (>= i), 'mid' is a possible end. Try larger j.
                if min_first_in_range >= i:
                    ans = mid
                    low = mid + 1
                else:
                    # If min_first < i, 'mid' ends too late. Need to end earlier.
                    high = mid - 1
            j_end[i] = ans
            
        # --- Step 5: Identify Candidate Special Substring Intervals ---
        # A substring s[i...j] is special if and only if it corresponds to an interval [i, j] where:
        # 1. i = min(first[c] for c in set(s[i..j]))
        # 2. j = max(last[c] for c in set(s[i..j]))
        # 3. length = j - i + 1 < n (not the entire string)
        # Conditions 1 and 2 are equivalent to i = i_start[j] AND j = j_end[i].
        # We iterate through potential end points 'j' and check consistency. Time O(n).
        candidate_intervals = []
        for j in range(n):
            i = i_start[j]
            # Check if a valid start index i (i <= j) was found for this j
            if i <= j:
                 # Check consistency: does this index i map back to j via j_end?
                 if j_end[i] == j:
                      # Check the special substring condition: must not be the entire string
                      if j - i + 1 < n:
                           # If all conditions met, [i, j] corresponds to a special substring
                           candidate_intervals.append((i, j))

        # --- Step 6: Find Maximum Number of Disjoint Intervals ---
        # This is the classic Activity Selection Problem: given a set of intervals, find the
        # maximum number of non-overlapping intervals.
        # Solved greedily after sorting the intervals by their end times.
        # Time: O(C log C + C), where C = number of candidates (C <= n). Overall O(n log n).
        
        # If no candidate intervals were found, we can't select k > 0 substrings.
        if not candidate_intervals:
             # Since k > 0 was handled at the start, this means k must be > 0 here.
            return False 

        # Sort the intervals based on their ending position (j)
        candidate_intervals.sort(key=lambda x: x[1])

        count = 0
        last_end = -1 # Stores the end position of the last selected interval, initialized to -1
        for i, j in candidate_intervals:
            # If the current interval's start (i) is strictly after the last selected interval's end,
            # then it's disjoint and can be selected.
            if i > last_end: 
                count += 1
                last_end = j # Update the end time of the last selected interval
        
        # --- Step 7: Final Result ---
        # Return true if the maximum number of disjoint special substrings found is at least k.
        return count >= k