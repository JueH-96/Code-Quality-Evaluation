class Solution:
  def minAnagramLength(self, s: str) -> int:
    n = len(s)
    
    # Calculate overall character counts for s.
    # This is used for an optimization (pre-filter for k).
    overall_counts_s = [0] * 26
    for char_in_s in s:
        overall_counts_s[ord(char_in_s) - ord('a')] += 1

    for k in range(1, n + 1):
        if n % k == 0:
            # k is a potential length for t.
            # If t has length k, s would be composed of m = n/k segments.
            m = n // k # Number of segments

            # Optimization: Pre-filter based on overall character counts.
            # If s is indeed formed by m anagrams of a string t (with length k),
            # then the total count of each character in s (overall_counts_s[char])
            # must be m times the count of that character in t (counts_t[char]).
            # This means overall_counts_s[char] must be divisible by m.
            
            possible_k_due_to_overall_counts = True
            for i in range(26): # For each character 'a' through 'z'
                if overall_counts_s[i] % m != 0:
                    possible_k_due_to_overall_counts = False
                    break
            
            if not possible_k_due_to_overall_counts:
                # If the divisibility check fails, this k cannot be the length of t.
                continue # Try the next potential value of k.

            # If the pre-filter passes, proceed to check segments.
            # The first segment s[0...k-1] must be an anagram of t.
            # We use its character counts as the reference for t's counts.
            first_segment_counts = [0] * 26
            for i in range(k): # Iterate s[0] through s[k-1]
                first_segment_counts[ord(s[i]) - ord('a')] += 1
            
            # Now, check if all subsequent segments are anagrams of the first segment
            # (i.e., have the same character counts as first_segment_counts).
            all_segments_match_first = True
            # Segments are s[0*k ... 1*k-1], s[1*k ... 2*k-1], ..., s[(m-1)*k ... m*k-1]
            # We've processed segment 0 (j=0) to get first_segment_counts.
            # Now check segments j=1 to m-1.
            for j in range(1, m): 
                current_segment_start_idx = j * k
                current_segment_counts = [0] * 26
                for i in range(k): # Iterate s[current_segment_start_idx + 0] through s[current_segment_start_idx + k-1]
                    char_in_segment = s[current_segment_start_idx + i]
                    current_segment_counts[ord(char_in_segment) - ord('a')] += 1
                
                # Compare counts of current segment with the first segment's counts
                if current_segment_counts != first_segment_counts:
                    all_segments_match_first = False
                    break # Mismatch found, this k is not valid.
            
            if all_segments_match_first:
                # If all segments are anagrams of the first one,
                # then this k is a valid length for t.
                # Since we are iterating k from 1 upwards, this is the minimum such k.
                return k
    
    # This part should be unreachable given the problem constraints:
    # "s, which is known to be a concatenation of anagrams of some string t."
    # This implies k=n is always a valid solution (s is one anagram of t=s).
    # The loop for k will eventually reach k=n and return n if no smaller k worked.
    return n # Fallback, though theoretically not needed.