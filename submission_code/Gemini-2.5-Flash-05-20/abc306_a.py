# YOUR CODE HERE
import sys

def solve():
    # Read N (though not strictly needed in Python, we read it as per problem format)
    N = int(sys.stdin.readline())
    # Read string S
    S = sys.stdin.readline().strip()

    # Use a list to store characters for efficient string building
    result_chars = []
    for char in S:
        result_chars.append(char)
        result_chars.append(char)

    # Join the list of characters to form the final string
    print("".join(result_chars))

if __name__ == '__main__':
    solve()