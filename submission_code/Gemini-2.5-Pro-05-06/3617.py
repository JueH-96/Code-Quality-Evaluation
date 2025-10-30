class Solution:
  def possibleStringCount(self, word: str) -> int:
    n = len(word)
    # According to constraints: 1 <= word.length <= 100.
    # So, n is guaranteed to be at least 1.

    # This 'count' starts at 1. This accounts for 'word' itself being a
    # possible original string, which corresponds to the case where Alice
    # made no long presses.
    count = 1
    
    i = 0
    while i < n:
      # Identify the current character for the run.
      # word[i] is the first character of this run.
      # We need to find how long this run of identical characters is.
      
      # Advance j to find the end of the run.
      # j will be the index of the first character AFTER the run.
      j = i
      while j < n and word[j] == word[i]: # word[i] is the character of the current run
        j += 1
      
      # The run is word[i ... j-1].
      # Its length is (j - 1) - i + 1 = j - i.
      run_length = j - i
      
      # If a run has length L (run_length) >= 2, it could have been
      # the result of a single long press event.
      # Specifically, Alice could have intended to type a string s = P G' S_suffix,
      # where G' is a sequence of k identical characters (word[i]) of length L',
      # with 1 <= L' < L. (P and S_suffix are prefix/suffix parts of word).
      # She then long-pressed the L'-th character of G' (the last char in G').
      # This single character was expanded into M = L - L' + 1 copies.
      # (The L'-1 characters before it in G', plus M copies, make up G, which has length L).
      # A long press requires the expansion M to be at least 2 characters.
      # M = L - L' + 1 >= 2  =>  L - L' >= 1  =>  L' <= L - 1.
      # This condition is satisfied by the range of L' (1 <= L' < L means L' is from 1 to L-1).
      #
      # The number of possible values for L' is (L-1) - 1 + 1 = L-1.
      # Each such L' defines a distinct original string s.
      # So, this run contributes (run_length - 1) additional possible original strings.
      if run_length >= 2:
        count += (run_length - 1)
        
      # Move the pointer i to the start of the next run.
      # The next run will start at index j.
      i = j
      
    return count