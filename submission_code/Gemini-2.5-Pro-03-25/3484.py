import sys 
# sys.setrecursionlimit(10**6) # This line is generally not needed for this problem but included as a common practice in competitive programming.

class Solution:
    """
    This class provides a method to find the lexicographically smallest string
    obtainable by swapping adjacent digits of the same parity at most once.
    """
    def getSmallestString(self, s: str) -> str:
        """
        Finds the lexicographically smallest string by performing at most one swap
        of adjacent digits with the same parity.

        The algorithm iterates through the string from left to right. It looks for the 
        first pair of adjacent digits `s[i]` and `s[i+1]` that satisfy two conditions:
        1. They have the same parity (both are even or both are odd).
        2. The left digit `s[i]` is greater than the right digit `s[i+1]`.

        If such a pair is found at index `i`, swapping them will result in a 
        lexicographically smaller string because the character at index `i` becomes smaller. 
        Since we want the overall lexicographically smallest string, performing this swap 
        at the earliest possible index `i` guarantees the optimal result. The problem statement 
        allows at most one swap, so after finding the first beneficial swap, we perform it 
        and stop searching.

        If the loop completes without finding any such pair, it means no swap can make the 
        string lexicographically smaller according to the rules, so the original string 
        is returned as it is already the smallest possible.

        Args:
          s: The input string consisting only of digits. 
             Constraints: 2 <= s.length <= 100.

        Returns:
          The lexicographically smallest string that can be obtained after performing 
          at most one swap of adjacent digits with the same parity.
        """
        n = len(s)
        # Convert the input string to a list of characters. 
        # Strings are immutable in Python, so we use a list to allow modification (swapping).
        s_list = list(s)
        
        # Iterate through the list of characters from the first element up to the second to last element.
        # The loop considers adjacent pairs (s_list[i], s_list[i+1]).
        for i in range(n - 1):
            # Get the integer values of the two adjacent digits.
            # We need the integer values to check their parity using the modulo operator.
            digit1 = int(s_list[i])
            digit2 = int(s_list[i+1])
            
            # Check if both digits have the same parity.
            # Parity is the remainder when divided by 2. 
            # (digit % 2) is 0 for even digits and 1 for odd digits.
            # If (digit1 % 2) == (digit2 % 2), they have the same parity.
            if (digit1 % 2) == (digit2 % 2):
                
                # If they have the same parity, check if swapping them would result in a 
                # lexicographically smaller string. This occurs if the character representation 
                # of the left digit is greater than that of the right digit.
                # For single digits '0' through '9', character comparison works correctly 
                # for numerical comparison (e.g., '5' > '3').
                if s_list[i] > s_list[i+1]:
                    
                    # Perform the swap of the two adjacent characters in the list.
                    # Python's tuple assignment makes swapping concise.
                    s_list[i], s_list[i+1] = s_list[i+1], s_list[i]
                    
                    # The problem states we can perform AT MOST one swap. 
                    # To achieve the lexicographically smallest result, we must perform the swap 
                    # at the leftmost possible position (smallest index `i`) where it is beneficial.
                    # Once we find this first beneficial swap and perform it, we must stop searching 
                    # for further swaps. The `break` statement exits the loop.
                    break
            
        # After the loop finishes (either by iterating through all pairs or by breaking after a swap),
        # join the characters in the list back into a single string.
        # If no swap was performed, the list remains unchanged, and joining it effectively returns the original string.
        return "".join(s_list)