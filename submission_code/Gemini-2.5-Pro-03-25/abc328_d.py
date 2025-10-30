# YOUR CODE HERE
import sys

# Define the main logic within a function for clarity
def solve():
    # Read the input string S from standard input. 
    # .strip() removes leading/trailing whitespace including the newline character.
    S = sys.stdin.readline().strip()
    
    # Initialize an empty list `res` which will be used as a stack.
    # This list will store the characters of the string being built,
    # after potentially removing "ABC" substrings.
    res = [] 
    
    # Iterate through each character `char` in the input string `S`.
    for char in S:
        # Append the current character `char` to the end of the list `res`.
        # This simulates pushing onto a stack.
        res.append(char)
        
        # Check if the list `res` contains at least 3 elements.
        # This check is necessary because we are looking for the substring "ABC", which has length 3.
        # If the list has fewer than 3 elements, it cannot end in "ABC".
        if len(res) >= 3:
            # Check if the last three elements in `res` are 'A', 'B', 'C' in that specific order.
            # `res[-1]` refers to the last element (top of the stack).
            # `res[-2]` refers to the second last element.
            # `res[-3]` refers to the third last element.
            if res[-3] == 'A' and res[-2] == 'B' and res[-1] == 'C':
                # If the last three elements form the sequence "ABC", 
                # we remove these three elements from the list `res`.
                # The `pop()` method removes and returns the last element of the list.
                # Calling `pop()` three times effectively removes the "ABC" sequence from the end of the list.
                res.pop() # Removes the last element ('C')
                res.pop() # Removes the new last element ('B')
                res.pop() # Removes the new last element ('A')
                
    # After iterating through all characters of the input string S,
    # the list `res` contains the characters of the final string according to the described process.
    # Join the characters in the list `res` together to form a single string.
    # The "".join(res) method concatenates all elements in the list `res` into a string.
    final_string = "".join(res)
    
    # Print the resulting final string to standard output.
    print(final_string)

# Call the solve function to execute the program logic.
# This ensures that the code runs when the script is executed.
solve()