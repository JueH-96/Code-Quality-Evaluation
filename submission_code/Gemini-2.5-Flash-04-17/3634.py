class Solution:
    def calculateScore(self, s: str) -> int:
        """
        Calculates the score based on matching characters with their mirrors.

        Args:
            s: The input string.

        Returns:
            The total score.
        """
        n = len(s)
        score = 0
        # We need to efficiently find the closest unmarked index j < i.
        # Since we iterate i from left to right, for a given character s[i],
        # the "closest" unmarked index j with the mirror character s[j]
        # will be the largest index j < i that is unmarked and has the mirror character.
        # We can maintain a list of available (unmarked) indices for each character.
        # As we iterate i, when s[i] requires its mirror s[j], we look for the
        # largest available index j in the list for the mirror character.
        # If the list is not empty, the last element is the largest index.
        # If a match (i, j) is made, both i and j are marked. j is removed
        # from its list. i is not added to its list.
        # If no match is made for i, i remains unmarked and becomes available
        # to be matched with a future i' > i. So i is added to its list.

        # Use a list of 26 lists to store available indices for each character
        # available_indices[0] will store indices for 'a', available_indices[1] for 'b', etc.
        available_indices = [[] for _ in range(26)]

        ord_a = ord('a')
        ord_z = ord('z')

        for i in range(n):
            current_char = s[i]
            # Get the 0-based index of the current character in the alphabet ('a' is 0, 'b' is 1, ...)
            current_char_idx = ord(current_char) - ord_a

            # Calculate the 0-based index of the mirror character.
            # If current_char_idx is k, its mirror is at index 25-k.
            # This is because ord(char1) + ord(char2) = ord('a') + ord('z') for mirrors.
            # ord(mirror_char) = ord('a') + ord('z') - ord(current_char)
            # mirror_char_idx = (ord('a') + ord('z') - ord(current_char)) - ord('a')
            # mirror_char_idx = ord('z') - ord(current_char)
            # mirror_char_idx = (ord('z') - ord_a) - (ord(current_char) - ord_a)
            # mirror_char_idx = 25 - current_char_idx
            mirror_char_idx = 25 - current_char_idx

            # Check if there are any available unmarked indices for the mirror character
            # The available_indices list for mirror_char_idx stores indices j < i
            # in increasing order because we iterate i from 0 to n-1.
            # The last element is the largest such index, which is the closest.
            if available_indices[mirror_char_idx]:
                # Found the closest available index j for the mirror character
                j = available_indices[mirror_char_idx].pop() # Get the largest index and remove it
                
                # Add the score (i - j)
                score += (i - j)

                # Indices i and j are now marked/used.
                # j was already removed from its list by .pop().
                # We do NOT add i to available_indices[current_char_idx]
                # because it has been marked by pairing with j.
            else:
                # No available unmarked index j < i found for the mirror character.
                # Index i was not matched with any j < i.
                # Index i is now available to be matched with a future i' > i.
                # Add the current index i to the list of available indices for current_char.
                # This list will be used when a future index i' > i is looking for current_char as its mirror.
                available_indices[current_char_idx].append(i)

        return score