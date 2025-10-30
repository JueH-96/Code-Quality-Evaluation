import sys

# It's good practice to put the core logic in a function.
def solve():
    # Read N and D from the first line of input
    # Using sys.stdin.readline for potentially faster I/O, common in competitive programming
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    D = int(line1[1])
    
    # Read the string S representing the initial state of boxes
    # .strip() is important to remove the trailing newline character from readline()
    S_str = sys.stdin.readline().strip()
    
    # Convert the string S to a list of characters, as strings are immutable in Python.
    # This allows us to modify the state of individual boxes.
    box_state = list(S_str)
    
    # Keep track of how many cookies Takahashi has eaten
    cookies_eaten_count = 0
    
    # Takahashi eats D cookies. On each day, he eats the cookie from the
    # rightmost box that contains a cookie. This implies he will eat the
    # D cookies that are located at the largest indices among all cookies initially.
    # We can find these by iterating from right to left over the boxes.
    
    # Iterate through the boxes from right to left (0-indexed, so from N-1 down to 0)
    for i in range(N - 1, -1, -1):
        # If the current box (at index i) contains a cookie
        if box_state[i] == '@':
            # Takahashi eats this cookie. So, the box becomes empty.
            box_state[i] = '.'
            # Increment the count of cookies eaten
            cookies_eaten_count += 1
            
            # If Takahashi has eaten D cookies, he has completed his D days of eating.
            # We can break out of the loop as no more cookies will be eaten.
            # The problem guarantees S contains at least D occurrences of '@',
            # so cookies_eaten_count will reach D.
            if cookies_eaten_count == D:
                break
                
    # After D cookies have been eaten, join the characters in box_state
    # to form the final string representation of the boxes.
    final_S_str = "".join(box_state)
    
    # Print the final state string to standard output.
    # sys.stdout.write requires a string and appending newline explicitly if needed for console output.
    sys.stdout.write(final_S_str + "
")

# This is the standard way to ensure solve() is called when the script is executed.
if __name__ == '__main__':
    solve()