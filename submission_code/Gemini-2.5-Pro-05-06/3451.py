class Solution:
  def compressedString(self, word: str) -> str:
    comp_parts = []
    n = len(word)
    i = 0  # This index tracks the current position in the input 'word'
    
    # Iterate through the word until all characters are processed
    while i < n:
      current_char = word[i]  # The character for the current segment
      count = 0               # Counter for the length of the current segment
      
      # This index, j_scan, is used to find the end of the current segment.
      # A segment consists of 'current_char' repeating up to 9 times.
      j_scan = i
      
      # Scan forward to count consecutive identical characters (current_char).
      # The loop stops if:
      # 1. The end of the word is reached (j_scan >= n).
      # 2. A different character is encountered (word[j_scan] != current_char).
      # 3. The count for the current segment reaches 9 (count == 9 implies next count would be 10, so count < 9 ensures it processes up to the 9th character).
      while j_scan < n and word[j_scan] == current_char and count < 9:
        count += 1
        j_scan += 1
      
      # Append the count (as a string) and the character to our list of parts
      comp_parts.append(str(count))
      comp_parts.append(current_char)
      
      # Move the main index 'i' forward by the number of characters processed
      # in this segment. This prepares 'i' for the next segment.
      i += count
      # Note: After the inner loop, j_scan = (initial i) + count.
      # So, 'i = i + count' is equivalent to setting the new 'i' to the current 'j_scan'.
      
    # Join all the collected parts (e.g., ["9", "a", "5", "a"]) into the final result string
    return "".join(comp_parts)