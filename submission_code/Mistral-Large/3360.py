class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter

        # Count the frequency of each character
        freq = Counter(word)

        # Convert the frequencies to a list and sort it
        frequencies = list(freq.values())
        frequencies.sort()

        # Initialize the number of deletions
        deletions = 0

        # Use two pointers to adjust the frequencies
        left, right = 0, len(frequencies) - 1

        while left < right:
            if frequencies[right] - frequencies[left] <= k:
                break
            # Decrease the higher frequency
            frequencies[right] -= 1
            deletions += 1
            # If the higher frequency becomes equal to the lower frequency, move the left pointer
            if frequencies[right] == frequencies[left]:
                left += 1
            # If the higher frequency becomes less than the lower frequency, move the right pointer
            if frequencies[right] < frequencies[left]:
                right -= 1

        return deletions