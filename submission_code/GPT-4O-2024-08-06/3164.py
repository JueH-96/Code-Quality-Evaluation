class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []  # List to store visited integers
        result = []  # List to store the result for each "prev"

        k = 0  # Counter for consecutive "prev" strings

        for word in words:
            if word == "prev":
                k += 1
                # Check if we have enough visited integers
                if k <= len(nums):
                    # Get the (k-1)th element from the end of nums
                    result.append(nums[-k])
                else:
                    # Not enough visited integers
                    result.append(-1)
            else:
                # Reset k as we encountered a new integer
                k = 0
                # Add the integer to the visited list
                nums.append(int(word))

        return result