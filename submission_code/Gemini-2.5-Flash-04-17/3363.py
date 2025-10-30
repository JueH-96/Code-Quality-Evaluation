from typing import List
from sortedcontainers import SortedDict

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        """
        Tracks the frequency of IDs and returns the count of the most frequent ID
        after each step.

        Args:
            nums: A list of integers representing the IDs.
            freq: A list of integers representing the change in frequency for
                  the corresponding ID in nums.

        Returns:
            A list of integers where ans[i] is the count of the most frequent ID
            after step i.
        """
        # counts: Stores the current count of each ID. {id: count}
        counts = {}

        # freq_counts: Stores the number of IDs that currently have a specific count.
        # {count: number_of_ids_with_this_count}
        # Using SortedDict allows efficient lookup of the largest count (max frequency).
        freq_counts = SortedDict()

        ans = []

        for i in range(len(nums)):
            id = nums[i]
            change = freq[i]

            # Get the old count of the current ID. Default to 0 if ID not seen before.
            old_count = counts.get(id, 0)
            # Calculate the new count.
            new_count = old_count + change

            # --- Update freq_counts based on the old count ---
            # If the old count was positive, it means there was at least one ID
            # with this specific frequency. We decrement the count for this frequency.
            if old_count > 0:
                freq_counts[old_count] -= 1
                # If the count for this frequency becomes 0, remove it from freq_counts
                # to keep the structure clean and efficient.
                if freq_counts[old_count] == 0:
                    del freq_counts[old_count]

            # --- Update the count for the current ID ---
            counts[id] = new_count

            # --- Update freq_counts based on the new count ---
            # Increment the count for the new frequency (new_count).
            # Use .get(new_count, 0) to handle cases where new_count is a frequency
            # seen for the first time.
            freq_counts[new_count] = freq_counts.get(new_count, 0) + 1

            # --- Determine the most frequent ID count after this step ---
            # The most frequent count is the largest key in freq_counts.
            # SortedDict stores keys in sorted order, so the last key is the largest.
            if freq_counts:
                # Get the largest frequency value (the key) from the SortedDict.
                # freq_counts.keys()[-1] gives the maximum count value currently present.
                # Example: If freq_counts is {0: 5, 2: 3}, keys()[-1] is 2.
                # Example: If freq_counts is {0: 5}, keys()[-1] is 0.
                current_max_freq = freq_counts.keys()[-1]
            else:
                # If freq_counts is empty, it means the collection is empty.
                current_max_freq = 0

            # Append the result for this step to the answer list.
            ans.append(current_max_freq)

        return ans