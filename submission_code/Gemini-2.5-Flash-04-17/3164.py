from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = [] # Stores integers encountered so far
        results = [] # Stores results for each "prev"
        consecutive_prev_count = 0 # Counts consecutive "prev" strings

        for word in words:
            if word == "prev":
                consecutive_prev_count += 1
                k = consecutive_prev_count # k is the 1-based count of consecutive "prev"s

                # We need the integer at index k-1 in the reversed list of nums.
                # This is the k-th integer from the end of the original nums list (1-indexed k).
                # In a 0-indexed list of length N, the k-th element from the end (1-indexed k)
                # is located at index N - k.
                target_index_in_nums = len(nums) - k

                # If k is greater than the total number of visited integers (len(nums)),
                # there is no k-th integer from the end. This corresponds to
                # the target index being negative.
                if target_index_in_nums >= 0:
                    # Valid index (k <= len(nums)), append the integer
                    results.append(nums[target_index_in_nums])
                else:
                    # Invalid index (k > len(nums)), append -1
                    results.append(-1)
            else:
                # The word is a string representation of a positive integer
                nums.append(int(word))
                # When we encounter a non-"prev" word, the consecutive count resets
                consecutive_prev_count = 0

        return results