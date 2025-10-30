import collections
import itertools

class Solution:
  def maximumLength(self, s: str) -> int:
    n = len(s) # s.length is guaranteed to be >= 3.
    
    # counts[char][length] stores the total number of occurrences of
    # the special substring formed by 'char' repeated 'length' times.
    # For example, counts['a'][2] = 5 means "aa" occurred 5 times in s.
    counts = collections.defaultdict(lambda: collections.defaultdict(int))

    # Part 1: Identify all runs of identical characters and their lengths.
    # A run is a consecutive sequence of identical characters, e.g., "aaa" or "bb".
    # We use itertools.groupby to find these runs efficiently.
    # For s = "aaabaaa", char_runs would be [('a', 3), ('b', 1), ('a', 3)],
    # representing a run of 'a's of length 3, then 'b's of length 1, then 'a's of length 3.
    char_runs = []
    for char_val, group_iter in itertools.groupby(s):
        char_runs.append((char_val, len(list(group_iter))))
        
    # Part 2: For each run, calculate its contribution to special substring counts.
    # A run of character 'x' of length k_run_len (e.g., "xxxx" so k_run_len=4)
    # contains multiple special substrings. For example, "xxxx" contains:
    # - "x" (length 1): occurs k_run_len - 1 + 1 = k_run_len times within this run.
    # - "xx" (length 2): occurs k_run_len - 2 + 1 = k_run_len - 1 times within this run.
    # ...
    # - "x" repeated l_sub_len times: occurs k_run_len - l_sub_len + 1 times.
    # We sum these contributions over all runs for each (char_val, l_sub_len) pair.
    for char_val, k_run_len in char_runs:
        for l_sub_len in range(1, k_run_len + 1):
            # Number of times substring of l_sub_len appears in this run of k_run_len
            num_occurrences_in_this_run = k_run_len - l_sub_len + 1
            counts[char_val][l_sub_len] += num_occurrences_in_this_run
    
    max_L = -1
    # Part 3: Find the maximum length L such that a special substring of length L
    # (composed of some character) occurs at least 3 times.
    # We iterate through the counts we've gathered.
    # counts.values() gives iterators over dictionaries like {1:4, 2:3, ...} for each character.
    for char_specific_counts in counts.values():
        # char_specific_counts is a map from length to total_occurrences for a specific character.
        # Example: for character 'a', it might be {1:4, 2:3}, meaning "a" occurred 4 times, "aa" 3 times.
        for length, num_total_occurrences in char_specific_counts.items():
            if num_total_occurrences >= 3:
                max_L = max(max_L, length)
            
    return max_L