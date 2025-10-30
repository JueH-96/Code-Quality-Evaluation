import itertools

class Solution:
  """
  Finds the minimum length string containing three given strings a, b, and c as substrings.
  If multiple strings have minimum length, returns the lexicographically smallest one.
  """
  def minimumString(self, a: str, b: str, c: str) -> str:
    
    # Helper function to combine two strings s1 and s2.
    # It computes the shortest string that starts with s1 and contains s2 as a substring,
    # maximizing the overlap where a suffix of s1 is a prefix of s2.
    # If s2 is already a substring of s1, it returns s1.
    def combine(s1: str, s2: str) -> str:
        """
        Computes the shortest string formed by concatenating s1 and s2 with maximum overlap,
        where s1 comes first. If s2 is a substring of s1, returns s1.
        """
        # If s2 is already contained within s1, then s1 itself is the shortest combined string
        # satisfying the condition that s1 is fully present, and s2 is present.
        if s2 in s1:
            return s1
        
        # Find the maximum length k >= 0 such that s1 ends with the prefix of s2 of length k.
        # This represents the longest overlap possible when s2 is appended to s1.
        max_k = 0
        # Iterate k downwards from the maximum possible overlap length to 1.
        # The first match found is the longest suffix of s1 that is also a prefix of s2.
        for k in range(min(len(s1), len(s2)), 0, -1):
            if s1.endswith(s2[:k]):
                max_k = k
                break
        
        # Append the part of s2 that does not overlap with the suffix of s1.
        # If no overlap (max_k=0), this appends the entire s2.
        return s1 + s2[max_k:]

    # Initial list of the three input strings.
    strings = [a, b, c]
    
    # Remove duplicate strings while preserving the order of first appearance.
    # Using dict.fromkeys is a standard Python trick for this.
    distinct_strings = list(dict.fromkeys(strings))
    
    # Filter the list further to remove any string that is a substring of another string in the list.
    # If string `s1` is a substring of `s2`, any superstring containing `s2` will automatically contain `s1`.
    # Thus, we only need to ensure the "maximal" strings (those not substrings of others) are included.
    final_strings = []
    for s1 in distinct_strings:
        is_substring = False
        for s2 in distinct_strings:
            # Check if s1 is a substring of s2, AND s1 is not identical to s2.
            # The s1 != s2 check prevents removing a string just because it's identical to another
            # (duplicates already removed) or if it's the only string left.
            if s1 != s2 and s1 in s2:
                is_substring = True
                break
        # If s1 was not found to be a substring of any other distinct string, add it to the final list.
        if not is_substring:
            final_strings.append(s1)

    # Based on problem constraints (non-empty input strings), final_strings will contain at least one string.
    # If final_strings happens to be empty (e.g., if inputs could be empty), handle appropriately.
    # However, with lengths >= 1, this part should always have at least one element.

    # Generate all possible orderings (permutations) of the strings in final_strings.
    # The order affects how strings are combined and the resulting superstring length and content.
    perms = list(itertools.permutations(final_strings))
    
    candidates = []
    
    # For each permutation, construct a candidate superstring.
    for p in perms:
        # Handle the case of a single string permutation (if k=1 string remains after filtering).
        if not p: # Check if permutation tuple is empty, although this shouldn't happen.
             continue

        # Start with the first string in the current permutation order.
        current_candidate = p[0]
        # Sequentially combine the remaining strings in the permutation order.
        # The combine function handles overlaps efficiently.
        for i in range(1, len(p)):
            current_candidate = combine(current_candidate, p[i])
        # Add the resulting superstring for this permutation to the list of candidates.
        candidates.append(current_candidate)

    # After generating candidates for all permutations, find the best one.
    # The best candidate has the minimum length. If there are ties in length,
    # choose the lexicographically smallest among them.
    # Sorting achieves this: sort primarily by length (ascending), secondarily by string value.
    candidates.sort(key=lambda x: (len(x), x))
    
    # The first element in the sorted list is the final answer.
    return candidates[0]