from collections import Counter
from sortedcontainers import SortedList

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        char_count = Counter(word)
        frequency_list = SortedList(char_count.values())
        total_deletions = 0

        while len(frequency_list) > 1:
            lowest_freq = frequency_list[0]
            highest_freq = frequency_list[-1]

            # Process highest frequency uniquely if it alone meets the criteria
            if len(frequency_list) == 2 and highest_freq - lowest_freq <= k:
                break

            # Reduce the highest frequency to make it comparable with the next highest
            reduce_by = min(highest_freq - lowest_freq + 1 - k, highest_freq)
            highest_freq -= reduce_by
            total_deletions += reduce_by
            frequency_list.remove(highest_freq + reduce_by)

            # Update the reduced frequency in the list if it's above 0
            if highest_freq > 0:
                idx = frequency_list.bisect_right(highest_freq) - 1
                if idx >= 0 and frequency_list[idx] == highest_freq:
                    frequency_list[idx] = highest_freq + 1
                frequency_list.add(highest_freq)

            # Remove the lowest frequency if it has become zero after processing
            if lowest_freq <= k:
                frequency_list.pop(0)

        # Calculate deletions for the final remaining elements if any
        if frequency_list and frequency_list[-1] > k:
            total_deletions += frequency_list[-1] - k

        return total_deletions