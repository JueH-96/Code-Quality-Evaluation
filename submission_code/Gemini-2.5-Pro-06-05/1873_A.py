# YOUR CODE HERE
import sys

def main():
    """
    This function reads all test cases and solves them one by one.
    """
    try:
        # Read the number of test cases.
        num_test_cases = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # Handle cases with no input or malformed input for robustness.
        num_test_cases = 0

    for _ in range(num_test_cases):
        # Read the 3-character string for the current test case.
        s = sys.stdin.readline().strip()
        
        # A string can be transformed to "abc" in at most one swap if and only if
        # at least one character is already in its correct position.
        # This is because:
        # - If 3 are correct, it's "abc" (0 swaps).
        # - If 1 is correct, the other two are swapped (1 swap).
        # - If 0 are correct, it's a 3-cycle needing 2 swaps.
        # - It's impossible to have exactly 2 correct in a permutation of 3 elements.
        
        # Check if any character is in its target position.
        if s[0] == 'a' or s[1] == 'b' or s[2] == 'c':
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()