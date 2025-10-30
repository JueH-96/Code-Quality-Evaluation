import collections

class Solution:
    """
    Solves the problem of counting substrings with exactly k consonants and all vowels.
    """
    def countOfSubstrings(self, word: str, k: int) -> int:
        """
        Counts the number of substrings of 'word' that contain every vowel ('a', 'e', 'i', 'o', 'u')
        at least once and exactly 'k' consonants.

        Args:
            word: The input string. Consists only of lowercase English letters. Length is between 5 and 250.
            k: The required number of consonants. Non-negative integer, 0 <= k <= word.length - 5.

        Returns:
            The total count of such substrings.

        The approach uses nested loops to iterate through all possible substrings.
        The outer loop defines the start index 'i', and the inner loop defines the end index 'j'.
        For each substring word[i..j], we maintain the count of distinct vowels and the count of consonants
        incrementally as we extend the substring to the right (increasing 'j').

        If a substring word[i..j] satisfies both conditions (contains all 5 vowels and has exactly k consonants),
        we increment the total count.

        An optimization is included: if a substring starting at 'i' already has 5 distinct vowels
        but the number of consonants exceeds k, then any further extension of this substring
        (i.e., word[i..l] where l > j) will also fail the consonant count condition (since it will
        still have 5 distinct vowels and at least as many consonants). In this case, we can break
        the inner loop early and move to the next starting index 'i+1'.

        Time Complexity: O(n^2), where n is the length of the word. The nested loops iterate through
                         O(n^2) potential substrings. Inside the inner loop, updating counts and
                         checking conditions takes O(1) time.
        Space Complexity: O(1), as the `vowel_counts` dictionary stores at most 5 keys (for the vowels).
                          The space used does not depend on the input size n.
        """
        n = len(word)
        count = 0
        vowels_set = {'a', 'e', 'i', 'o', 'u'}

        # Iterate through all possible start indices 'i' of the substring
        for i in range(n):
            # Initialize counts for the substring starting at index i.
            # These counts will be updated as we extend the substring to the right (increase 'j').
            vowel_counts = collections.defaultdict(int) # Tracks counts of each vowel within the current substring word[i..j]
            consonant_count = 0                       # Tracks the number of consonants in word[i..j]
            num_distinct_vowels = 0                   # Tracks the number of unique vowels found in word[i..j]

            # Iterate through all possible end indices 'j' for the substring starting at 'i'
            for j in range(i, n):
                # Consider the character being added to the substring at the end
                char = word[j]
                is_vowel = char in vowels_set

                # Update the counts based on the newly added character word[j]
                if is_vowel:
                    # If this is the first time we encounter this vowel in the substring word[i..j]
                    if vowel_counts[char] == 0:
                        num_distinct_vowels += 1
                    vowel_counts[char] += 1
                else:
                    # It's a consonant
                    consonant_count += 1

                # Check if the current substring word[i..j] meets the required criteria:
                # 1. Contains all 5 distinct vowels (num_distinct_vowels == 5)
                # 2. Contains exactly k consonants (consonant_count == k)
                if num_distinct_vowels == 5 and consonant_count == k:
                    # If both conditions are met, increment the total count
                    count += 1

                # Optimization:
                # Check if we can break the inner loop early.
                # If we have already found all 5 distinct vowels (meaning num_distinct_vowels will remain 5 for any further extension j'>j),
                # and the number of consonants has strictly exceeded k,
                # then any substring word[i..l] where l > j will also have consonant_count > k.
                # Such substrings cannot satisfy the condition of having exactly k consonants.
                # Therefore, no further substring starting at 'i' can satisfy the conditions.
                # We can break the inner loop (over j) and move to the next starting index 'i+1'.
                # This check is placed after the count increment to ensure substrings ending at j
                # where consonant_count == k are counted.
                if num_distinct_vowels == 5 and consonant_count > k:
                    break

        # Return the total count of qualifying substrings found
        return count