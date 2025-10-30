import sys

def solve():
    # Read N from standard input
    N = int(sys.stdin.readline())

    # Initialize a list of N '-' characters
    result_chars = ['-'] * N

    # Determine placement of '=' based on N's parity
    if N % 2 == 1:
        # N is odd: exactly one '=' in the middle
        mid_idx = N // 2
        result_chars[mid_idx] = '='
    else:
        # N is even: exactly two adjacent '='s in the middle
        # The two middle indices are N/2 - 1 and N/2
        mid_right_idx = N // 2
        mid_left_idx = mid_right_idx - 1
        result_chars[mid_left_idx] = '='
        result_chars[mid_right_idx] = '='
    
    # Join the characters to form the final string and print to standard output
    sys.stdout.write("".join(result_chars) + "
")

# Call the solve function when the script is executed
if __name__ == '__main__':
    solve()