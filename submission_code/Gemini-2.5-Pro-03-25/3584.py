import bisect
import math
from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        """
        Finds the lexicographically smallest sequence of indices from word1 such that
        the concatenated characters form a string almost equal to word2.
        """
        n1 = len(word1)
        n2 = len(word2)

        # Precompute positions of each character in word1 for efficient lookups.
        # pos[i] stores a sorted list of indices where character corresponding to i (0='a', 1='b', etc.) appears in word1.
        pos = [[] for _ in range(26)]
        for i, char in enumerate(word1):
            pos[ord(char) - ord('a')].append(i)

        # Helper function using binary search (bisect_left):
        # Find the smallest index k >= min_idx such that word1[k] == target_char
        def find_next_idx(target_char_ord, min_idx):
            """Finds smallest index k >= min_idx where word1[k] has target_char_ord."""
            char_indices = pos[target_char_ord]
            # bisect_left finds the insertion point for min_idx, which is the index of the
            # first element in char_indices that is >= min_idx.
            p = bisect.bisect_left(char_indices, min_idx)
            # If p is within the bounds of the list, char_indices[p] is the desired index.
            if p < len(char_indices):
                return char_indices[p]
            # Otherwise, no such index exists. Return infinity as a sentinel value.
            return math.inf

        # Helper function: Find the smallest index k >= min_idx such that word1[k] != forbidden_char
        def find_next_idx_mismatch(forbidden_char_ord, min_idx):
            """Finds smallest index k >= min_idx where word1[k] is not forbidden_char."""
            min_k = math.inf
            # Iterate through all possible characters (0-25).
            for char_ord in range(26):
                # Skip the forbidden character.
                if char_ord == forbidden_char_ord:
                    continue
                # For each allowed character, find the smallest index k >= min_idx.
                k = find_next_idx(char_ord, min_idx)
                # Keep track of the minimum k found among all allowed characters.
                min_k = min(min_k, k)
            return min_k

        # Backward Dynamic Programming phase:
        # Compute min_start_idx[j][d], which stores the minimum index i_j in word1
        # such that it's possible to start matching the suffix word2[j:] from index i_j,
        # using exactly d differences for this suffix match.
        # This DP provides feasibility information for the greedy construction phase.
        min_start_idx = [[math.inf] * 2 for _ in range(n2 + 1)]
        # Base case: After matching all of word2 (at index n2), the conceptual next index is n1.
        # Matching an empty suffix requires 0 differences.
        min_start_idx[n2][0] = n1 

        # Helper: Find smallest index k such that word1[k] == target_char AND k < upper_bound_exclusive.
        # This checks if the first occurrence of target_char is less than the upper_bound. Runs in O(1).
        def find_min_idx_strict_lt(target_char_ord, upper_bound_exclusive):
             """Finds smallest index k < upper_bound_exclusive where word1[k] has target_char_ord."""
             char_indices = pos[target_char_ord]
             # If no indices exist for this character, or the smallest index is already >= upper_bound, return infinity.
             if not char_indices or char_indices[0] >= upper_bound_exclusive:
                 return math.inf
             # The overall smallest index for this character satisfies k < upper_bound.
             return char_indices[0]
        
        # Helper: Find smallest index k such that word1[k] != forbidden_char AND k < upper_bound_exclusive.
        # Takes O(alphabet_size) time.
        def find_min_idx_mismatch_strict_lt(forbidden_char_ord, upper_bound_exclusive):
             """Finds smallest index k < upper_bound_exclusive where word1[k] is not forbidden_char."""
             min_k = math.inf
             # Iterate through all characters except the forbidden one.
             for char_ord in range(26):
                 if char_ord == forbidden_char_ord:
                     continue
                 # Find the smallest index k < upper_bound for this allowed character.
                 k = find_min_idx_strict_lt(char_ord, upper_bound_exclusive)
                 min_k = min(min_k, k)
             return min_k

        # DP calculation loop: Fill the min_start_idx table backwards from j = n2-1 down to 0.
        for j in range(n2 - 1, -1, -1):
            target_char_ord = ord(word2[j]) - ord('a')
            
            # Calculate min_start_idx[j][0]: Need to match word2[j] correctly (0 diffs).
            # Requires finding smallest k such that word1[k] == word2[j] AND k < min_start_idx[j+1][0].
            # The condition k < min_start_idx[j+1][0] ensures that there's room for the rest of the sequence
            # starting at index min_start_idx[j+1][0] or later.
            next_min_req_0 = min_start_idx[j + 1][0]
            if next_min_req_0 != math.inf:
                 k0 = find_min_idx_strict_lt(target_char_ord, next_min_req_0)
                 if k0 != math.inf:
                     min_start_idx[j][0] = k0

            # Calculate min_start_idx[j][1]: Need to match word2[j:] using exactly 1 diff.
            # Path 1: Match word2[j] correctly (0 diffs here), use 1 diff for word2[j+1:].
            # Requires smallest k1 such that word1[k1] == word2[j] AND k1 < min_start_idx[j+1][1].
            next_min_req_1 = min_start_idx[j + 1][1]
            k1 = math.inf
            if next_min_req_1 != math.inf:
                 k1 = find_min_idx_strict_lt(target_char_ord, next_min_req_1)
            
            # Path 2: Match word2[j] incorrectly (1 diff here), use 0 diffs for word2[j+1:].
            # Requires smallest k2 such that word1[k2] != word2[j] AND k2 < min_start_idx[j+1][0].
            next_min_req_0 = min_start_idx[j + 1][0] # Fetch value again
            k2 = math.inf
            if next_min_req_0 != math.inf:
                k2 = find_min_idx_mismatch_strict_lt(target_char_ord, next_min_req_0)

            # The minimum start index for d=1 is the minimum of the possibilities from Path 1 and Path 2.
            min_start_idx[j][1] = min(k1, k2)

        # Greedy construction phase: Build the result sequence step by step.
        result = []
        current_diffs = 0 # Tracks if the one allowed difference has been used (0: not used, 1: used).
        last_idx = -1     # Tracks the index in word1 used for the previous character match.

        # Check overall feasibility: If it's impossible to even start matching word2[0], return [].
        if min(min_start_idx[0][0], min_start_idx[0][1]) == math.inf:
            return []

        # Iterate through word2 to find the corresponding indices in word1.
        for j in range(n2):
            target_char_ord = ord(word2[j]) - ord('a')
            
            match_feasible_k = math.inf      # Stores the best index found via matching path.
            mismatch_feasible_k = math.inf   # Stores the best index found via mismatching path.

            # Option 1: Try matching word2[j] correctly using word1[k] where k > last_idx.
            k_match = find_next_idx(target_char_ord, last_idx + 1)
            if k_match != math.inf:
                 # Check feasibility: Is it possible to complete the match for word2[j+1:] 
                 # starting from an index strictly greater than k_match?
                 possible = False
                 if current_diffs == 0:
                     # If no difference used yet, suffix can use 0 or 1 difference.
                     # Check if the minimum required start index for suffix (either 0 or 1 diff) is > k_match.
                     req0 = min_start_idx[j + 1][0]
                     req1 = min_start_idx[j + 1][1]
                     if req0 > k_match or req1 > k_match:
                         possible = True
                 else: # current_diffs == 1
                     # If difference already used, suffix must use 0 differences.
                     # Check if the minimum required start index for 0-diff suffix is > k_match.
                     req0 = min_start_idx[j + 1][0]
                     if req0 > k_match:
                         possible = True
                 
                 if possible:
                     match_feasible_k = k_match

            # Option 2: Try matching word2[j] incorrectly using word1[k] (only if difference not used yet).
            if current_diffs == 0:
                k_mismatch = find_next_idx_mismatch(target_char_ord, last_idx + 1)
                if k_mismatch != math.inf:
                     # Check feasibility: Suffix word2[j+1:] must be matched with 0 differences,
                     # starting from an index strictly greater than k_mismatch.
                     req0 = min_start_idx[j + 1][0]
                     if req0 > k_mismatch:
                         mismatch_feasible_k = k_mismatch

            # Choose the best feasible option for index i_j based on lexicographical order.
            final_k = math.inf
            used_mismatch = False
            
            # Prefer the smaller index k. If indices are equal, prefer the match path (uses 0 diffs at this step).
            if match_feasible_k <= mismatch_feasible_k: 
                 final_k = match_feasible_k
                 # used_mismatch remains False
            elif mismatch_feasible_k < match_feasible_k: # This condition implies mismatch_feasible_k is finite.
                 final_k = mismatch_feasible_k
                 used_mismatch = True
            
            # If neither path yielded a feasible index, no valid sequence exists.
            if final_k == math.inf:
                 return [] 

            # Append the chosen index to the result sequence.
            result.append(final_k)
            # Update the last used index.
            last_idx = final_k
            # If mismatch was used, update the difference counter.
            if used_mismatch:
                current_diffs = 1 

        # If the loop completes, a valid sequence has been found.
        return result