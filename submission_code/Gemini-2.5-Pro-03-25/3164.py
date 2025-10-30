from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        """
        Calculates the last visited integers based on "prev" commands in a list of strings.

        The function iterates through the input list `words`. It maintains a list `nums` 
        of integers encountered so far and a counter `k` for consecutive "prev" strings.
        When an integer string is found, it's converted to an integer, added to `nums`, 
        and `k` is reset to 0.
        When "prev" is found, `k` is incremented. The function then attempts to find the 
        k-th last integer added to `nums`. This corresponds to the element at index `len(nums) - k`.
        If `k` is greater than the number of integers currently in `nums`, -1 is recorded.
        Otherwise, the found integer is recorded. The recorded integers for each "prev" 
        operation form the output list.

        Args:
            words: A list of strings, where each element is either a positive
                   integer represented as a string (e.g., "1", "100") or the 
                   string "prev".

        Returns:
            A list of integers. Each integer in the list corresponds to a "prev" 
            string in the input `words`. It represents the k-th last visited integer,
            where k is the number of consecutive "prev" strings ending at that point.
            If the requested k-th last integer does not exist (k > number of visited integers),
            -1 is returned for that "prev".
        """
        
        nums = []  # Stores the sequence of integers encountered.
        result = [] # Stores the output corresponding to each "prev" command.
        k = 0      # Counts the number of consecutive "prev" commands seen.

        # Iterate through each word in the input list.
        for word in words:
            if word == "prev":
                # If the word is "prev", increment the consecutive counter.
                k += 1 
                
                # Check if the k-th last visited integer exists.
                # The number of integers visited so far is len(nums).
                if k > len(nums):
                    # If k is greater than the number of integers seen, 
                    # the k-th last integer doesn't exist. Append -1 to the result.
                    result.append(-1)
                else:
                    # Otherwise, the k-th last integer exists.
                    # It is located at index `len(nums) - k` in the `nums` list.
                    # (Since list indices are 0-based, the last element is at len(nums)-1, 
                    # the second last at len(nums)-2, and the k-th last at len(nums)-k).
                    last_visited = nums[len(nums) - k]
                    result.append(last_visited)
            else:
                # If the word is not "prev", it must be an integer string (based on constraints).
                # Convert the string to an integer.
                num_val = int(word)
                # Add the integer to the list of visited integers.
                nums.append(num_val)
                # Reset the consecutive "prev" counter because we encountered an integer.
                k = 0 
        
        # Return the list containing the results for each "prev" command.
        return result