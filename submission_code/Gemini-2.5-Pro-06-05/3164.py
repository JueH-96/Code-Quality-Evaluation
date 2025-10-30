class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        """
        Processes a list of strings, which are either numbers or "prev",
        and returns the last visited integers for each "prev" command.
        """
        # A list to store the integers we have seen so far.
        nums = []
        
        # k will count the number of consecutive "prev" strings.
        k = 0
        
        # The list to store the results for each "prev".
        ans = []
        
        for word in words:
            if word == "prev":
                # Increment the consecutive "prev" counter.
                k += 1
                
                # Check if the k-th last visited integer exists.
                if k > len(nums):
                    ans.append(-1)
                else:
                    # The k-th last visited integer is defined as the (k-1)th
                    # element of the reversed list of seen numbers. This can
                    # be accessed with `nums[-k]` using Python's negative indexing.
                    ans.append(nums[-k])
            else:
                # The current word is a number, so it breaks any "prev" sequence.
                # Reset the consecutive "prev" counter.
                k = 0
                
                # Convert the string to an integer and add it to our list.
                nums.append(int(word))
                
        return ans