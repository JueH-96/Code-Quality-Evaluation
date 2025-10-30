import collections

class Solution:
  def countOfSubstrings(self, word: str, k: int) -> int:
    n = len(word)
    ans = 0
    
    # Set of vowel characters for O(1) average time lookup
    vowel_chars = {'a', 'e', 'i', 'o', 'u'}

    # Outer loop: Iterate through all possible start indices 'i' for substrings
    for i in range(n):
        # For each starting index 'i', reset the counts for the new window
        
        # current_vowel_counts maps each vowel type (e.g., 'a') to its
        # frequency in the current window word[i..j].
        # collections.defaultdict(int) initializes counts to 0 for new keys.
        current_vowel_counts = collections.defaultdict(int)
        
        # current_distinct_vowel_count stores the number of unique vowel types
        # (e.g., if 'a' and 'e' are present, count is 2) in the window word[i..j].
        # This needs to reach 5 for the "every vowel" condition.
        current_distinct_vowel_count = 0
        
        # current_consonant_count stores the number of consonants in word[i..j].
        # This needs to be exactly 'k'.
        current_consonant_count = 0
        
        # Inner loop: Iterate through all possible end indices 'j' for substrings
        # starting at 'i'. The current substring being considered is word[i..j].
        for j in range(i, n):
            char = word[j] # The character being added to the window by extending to index j
            
            if char in vowel_chars:
                # If this specific vowel type (e.g., 'a') is encountered for the
                # first time in the current window word[i..j] (i.e., its count was 0),
                # then we've found one more type of distinct vowel.
                if current_vowel_counts[char] == 0:
                    current_distinct_vowel_count += 1
                current_vowel_counts[char] += 1 # Increment frequency for this vowel type
            else: # Character is a consonant
                current_consonant_count += 1
            
            # Optimization:
            # If the number of consonants in word[i..j] already exceeds k,
            # then any further extension of this window (word[i..j+1], word[i..j+2], etc.)
            # will also have more than k consonants (since consonant counts are non-decreasing).
            # Such windows cannot satisfy the "exactly k consonants" condition.
            # So, we can break from this inner loop (over j) and move to the next start 'i'.
            if current_consonant_count > k:
                break
            
            # Check if the current substring word[i..j] satisfies both required conditions:
            # 1. Contains every vowel ('a', 'e', 'i', 'o', 'u') at least once.
            #    This is true if current_distinct_vowel_count is 5.
            # 2. Contains exactly k consonants.
            #    This is true if current_consonant_count is equal to k.
            #    (Note: current_consonant_count <= k must be true at this point due to the break condition above.
            #    So, this check simplifies to current_consonant_count == k.)
            if current_distinct_vowel_count == 5 and current_consonant_count == k:
                ans += 1
                
    return ans