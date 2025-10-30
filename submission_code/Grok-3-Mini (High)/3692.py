import bisect

class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        def find_all_starts(sub):
            if not sub:
                return list(range(len(s) + 1))
            starts = []
            pos = 0
            while True:
                found = s.find(sub, pos)
                if found == -1:
                    break
                starts.append(found)
                pos = found + 1  # Handle overlapping occurrences
            return starts
        
        # Find the indices of the two '*' characters
        idx1 = p.index('*')
        idx2 = p.index('*', idx1 + 1)
        prefix_str = p[:idx1]
        middle_str = p[idx1 + 1 : idx2]
        suffix_str = p[idx2 + 1 :]
        len_prefix = len(prefix_str)
        len_middle = len(middle_str)
        len_suffix = len(suffix_str)
        
        # Get the start positions for prefix, middle, and suffix
        A = find_all_starts(prefix_str)
        B = find_all_starts(middle_str)
        C = find_all_starts(suffix_str)
        
        # Initialize answer to infinity
        ans = float('inf')
        
        # Iterate over each occurrence of the middle part
        for j in range(len(B)):
            # Calculate the upper value for A[i]
            upper_val = B[j] - len_prefix
            # Find the rightmost index i such that A[i] <= upper_val
            pos_i = bisect.bisect_right(A, upper_val)
            if pos_i > 0:
                i_idx = pos_i - 1
                A_pos = A[i_idx]  # Start position of prefix
                # Calculate the lower value for C[k]
                lower_val_C = B[j] + len_middle
                # Find the leftmost index k such that C[k] >= lower_val_C
                k_idx = bisect.bisect_left(C, lower_val_C)
                if k_idx < len(C):
                    C_pos = C[k_idx]  # Start position of suffix
                    # Calculate the length of the substring
                    len_val = C_pos - A_pos + len_suffix
                    # Update the minimum answer
                    ans = min(ans, len_val)
        
        # If no valid substring found, return -1; otherwise, return the minimum length
        return -1 if ans == float('inf') else ans