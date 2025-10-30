import itertools

class Solution:
  def minimumString(self, a: str, b: str, c: str) -> str:
    """
    Finds the minimum length string that contains a, b, and c as substrings.
    If multiple such strings exist, returns the lexicographically smallest one.
    """
    
    # Helper to merge s1 and s2 in a fixed order (s1 ... s2), returning the shortest superstring.
    def merge(s1: str, s2: str) -> str:
      if s2 in s1:
        return s1
      
      # Find the longest suffix of s1 that is a prefix of s2.
      # Iterate from the longest possible overlap down to no overlap.
      for i in range(min(len(s1), len(s2)), -1, -1):
        if s1.endswith(s2[:i]):
          return s1 + s2[i:]
      # This part is theoretically unreachable due to the loop condition i=0.
      return s1 + s2

    # Helper to find the shortest superstring for two strings by trying both merge orders.
    def combine(s1: str, s2: str) -> str:
      res1 = merge(s1, s2)
      res2 = merge(s2, s1)
      
      # Return the better result (shorter, or lexicographically smaller if same length).
      if len(res1) < len(res2):
        return res1
      elif len(res2) < len(res1):
        return res2
      else:
        return min(res1, res2)

    # Step 1: Filter the initial strings to get a minimal set.
    initial_strings = list(set([a, b, c]))
    
    strings_to_combine = []
    for s1 in initial_strings:
      is_substring = False
      for s2 in initial_strings:
        if s1 != s2 and s1 in s2:
          is_substring = True
          break
      if not is_substring:
        strings_to_combine.append(s1)
    
    if not strings_to_combine:
        # This case occurs if one string contains all others.
        # The shortest of the initial unique strings is the answer.
        return min(initial_strings, key=lambda s: (len(s), s))

    best_str = ""

    # Step 2: Iterate through all permutations of the filtered strings.
    for p in itertools.permutations(strings_to_combine):
      # Combine strings sequentially according to the current permutation.
      current_res = p[0]
      for i in range(1, len(p)):
        current_res = combine(current_res, p[i])
      
      # Step 3: Check if this candidate is better than the best one found so far.
      if not best_str or len(current_res) < len(best_str) or \
         (len(current_res) == len(best_str) and current_res < best_str):
        best_str = current_res
            
    return best_str