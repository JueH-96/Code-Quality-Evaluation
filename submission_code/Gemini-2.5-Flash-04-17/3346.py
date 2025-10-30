class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        # Convert the string to a list of characters for easier modification
        t_chars = list(s)

        # Iterate through the string from left to right
        for i in range(n):
            # If we have no budget left, we can stop changing characters
            if k == 0:
                break

            current_char = s[i] # Consider the original character at this position
            char_val = ord(current_char) - ord('a')

            # Calculate the minimum cost to change the current character to 'a'
            # using the cyclic distance definition.
            # The distance is the minimum of moving forward or backward on the alphabet cycle.
            # Moving backward from current_char to 'a' takes char_val steps (e.g. 'c'->'a' is 2 steps).
            # Moving forward from current_char to 'a' takes 26 - char_val steps (e.g. 'z'->'a' is 1 step forward).
            cost_to_a = min(char_val, 26 - char_val)

            # If we have enough budget to change the current character to 'a'
            if k >= cost_to_a:
                # Change the character to 'a'
                t_chars[i] = 'a'
                # Deduct the cost from the budget
                k -= cost_to_a
            else:
                # We don't have enough budget (k) to reach 'a' (the lexicographically smallest character)
                # using the shortest path (cost_to_a).
                # We have k budget remaining in total. To make the current character
                # t_chars[i] as small as possible using at most k budget, we should
                # move the original character s[i] k steps backward cyclically.
                # This uses exactly k budget and results in the smallest character
                # reachable with exactly k steps. The distance between s[i] and the
                # new character will be <= k.

                # Calculate the new character value after moving k steps backward cyclically.
                # original character value is char_val (0-25).
                # Moving k steps backward means the new value is char_val - k.
                # We take modulo 26 to handle cyclic wrap-around.
                # In Python, `(char_val - k) % 26` correctly handles negative intermediate results.
                new_char_val = (char_val - k) % 26
                
                # Convert the new character value back to a character
                t_chars[i] = chr(new_char_val + ord('a'))
                
                # We have used the entire remaining budget k for this character.
                k = 0 # Budget is now 0.

                # Since k is 0, we can make no further changes to the remaining characters.
                # The characters from index i+1 onwards in t_chars are already the same
                # as the original string s, which is correct because we can't change them anymore.
                # So we can break the loop.
                break

        # Join the list of characters back into a string
        return "".join(t_chars)