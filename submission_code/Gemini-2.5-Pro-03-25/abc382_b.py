# YOUR CODE HERE
import sys

def solve():
    """
    Solves the problem of determining the final state of cookie boxes after D days.
    Takahashi eats one cookie per day from the rightmost box containing a cookie.
    This is equivalent to removing the D cookies that were initially in the
    D rightmost positions among all cookies.
    """
    
    # Read input N (number of boxes) and D (number of days/cookies eaten)
    # sys.stdin.readline() reads a line from standard input including the newline character.
    # .split() splits the line into a list of strings based on whitespace.
    # map(int, ...) converts each string in the list to an integer.
    n, d = map(int, sys.stdin.readline().split())
    
    # Read the initial state of the boxes as a string.
    # .strip() removes leading/trailing whitespace, including the newline character.
    # list(...) converts the string into a list of characters, which is mutable.
    s_list = list(sys.stdin.readline().strip())

    # Find the indices of all boxes initially containing a cookie ('@')
    cookie_indices = []
    # Iterate through the boxes using their indices (0 to N-1)
    for i in range(n):
        # Check if the box at index i contains a cookie
        if s_list[i] == '@':
            # If it does, add its index to the list
            cookie_indices.append(i)

    # Calculate the total number of cookies initially present
    num_cookies = len(cookie_indices)

    # Determine the indices of the D cookies that will be eaten.
    # Since Takahashi always eats the rightmost available cookie, over D days,
    # he will eat the D cookies that were initially located at the D rightmost positions.
    # The `cookie_indices` list stores the indices of cookies in increasing order (from left to right).
    # Therefore, the indices of the D rightmost cookies are the last D elements of this list.
    # We use list slicing to get these last D indices.
    # The problem guarantees that num_cookies >= D, so `num_cookies - d` is non-negative
    # and the slice `[num_cookies - d:]` is valid.
    indices_to_eat = cookie_indices[num_cookies - d:]

    # Update the state of the boxes.
    # For each index corresponding to a cookie that is eaten,
    # change the character in the `s_list` at that index from '@' to '.'.
    for index in indices_to_eat:
        s_list[index] = '.'

    # Join the characters in the modified list back into a single string.
    final_state = "".join(s_list)
    
    # Print the resulting string representing the final state of the boxes to standard output.
    print(final_state)

# Execute the solve function to run the program when the script is executed.
# This is standard practice in competitive programming and standalone scripts.
if __name__ == '__main__':
    solve()