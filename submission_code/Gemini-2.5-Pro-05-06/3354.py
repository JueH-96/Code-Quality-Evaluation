import heapq

class Solution:
  def minimizeStringValue(self, s: str) -> str:
    # Step 1: Initialize character counts and find indices of '?'
    # char_counts[j] will store the count of character ('a' + j)
    char_counts = [0] * 26 
    
    q_indices = [] # Store indices of '?' characters
    s_list = list(s) # Convert string to list of characters for in-place modification
    
    for i, char_val in enumerate(s_list):
      if char_val == '?':
        q_indices.append(i)
      else:
        char_counts[ord(char_val) - ord('a')] += 1
        
    # Step 2: Prepare a min-priority queue for character selection
    # Each element in PQ is a tuple (count, char_idx)
    # char_idx is 0 for 'a', 1 for 'b', ..., 25 for 'z'.
    # heapq pops the element with the smallest count.
    # If counts are equal, it pops the one with the smallest char_idx (e.g., 'a' before 'b').
    # This tie-breaking is inherent to heapq when items are tuples and first elements are equal.
    pq = []
    for i in range(26):
      heapq.heappush(pq, (char_counts[i], i))
      
    # Step 3: Determine the multiset of characters to replace '?'s
    # These are collected in `replacement_chars`.
    # This list is built by repeatedly picking the character `c` that:
    #   1. has the minimum current frequency (`count`).
    #   2. is lexicographically smallest among those satisfying (1) (due to `char_idx` tie-breaking in PQ).
    # This ensures the multiset of selected characters is "as lexicographically small as possible",
    # which is essential for the overall lexicographically smallest final string among those with minimum value.
    replacement_chars = []
    num_q = len(q_indices) # Total number of '?'s
    
    for _ in range(num_q):
      count, char_idx = heapq.heappop(pq)
      # Add the chosen character to our list of characters for '?'s
      replacement_chars.append(chr(ord('a') + char_idx))
      
      # Increment the count for the chosen character and add it back to the PQ
      # This character will now be considered with its new (higher) count for future '?'s
      heapq.heappush(pq, (count + 1, char_idx))
      
    # Step 4: Sort the replacement characters alphabetically.
    # This is crucial for achieving the lexicographically smallest result string.
    # When filling '?'s (which occur at q_indices locations) from left to right, 
    # we must use the smallest available characters from our chosen multiset `replacement_chars`.
    replacement_chars.sort()
    
    # Step 5: Fill the '?' positions in s_list
    # q_indices contains the original positions of '?'s. These indices are already sorted
    # because they were collected by iterating through `s` from left to right.
    # So, s_list[q_indices[i]] refers to the i-th '?' from the left in the original string.
    # It should be replaced by the i-th character from the sorted `replacement_chars` list.
    for i in range(num_q):
      s_list[q_indices[i]] = replacement_chars[i]
      
    # Step 6: Convert the list of characters back to a string and return it.
    return "".join(s_list)