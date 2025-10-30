import sys

def solve():
    # Read N and D from the first line of input
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    D = int(line1[1])

    # Read the string S representing the initial state of boxes
    S = sys.stdin.readline().strip()

    # Convert the string S into a mutable list of characters.
    # This allows us to modify the state of the boxes (eat cookies).
    box_state = list(S)

    # Simulate the cookie-eating process for D days
    for _ in range(D):
        # On each day, Takahashi eats the rightmost cookie.
        # To find the rightmost cookie, iterate through the boxes from right to left.
        for i in range(N - 1, -1, -1): # i goes from N-1 down to 0
            if box_state[i] == '@':
                # If a cookie ('@') is found at index i, Takahashi eats it.
                # This means changing the box state from '@' to '.'
                box_state[i] = '.'
                # Once a cookie is eaten, the day ends, and we move to the next day.
                # So, break out of this inner loop.
                break
    
    # After D days, join the list of characters back into a string
    # to represent the final state of the boxes.
    final_state_str = "".join(box_state)

    # Print the resulting string
    print(final_state_str)

# Call the solve function to execute the program
if __name__ == '__main__':
    solve()