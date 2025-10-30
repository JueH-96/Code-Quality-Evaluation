# YOUR CODE HERE
import sys

def main():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Read problem inputs from standard input.
        N = int(sys.stdin.readline())
        S = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Graceful exit on malformed input.
        return

    # Constraints state 1 <= N, so N=0 check is for robustness.
    if N == 0:
        print(0)
        return

    # --- Step 1: Precomputation ---

    # `left_ones[i]` stores the number of consecutive '1's ending at index i.
    # This is calculated with a single pass from left to right.
    left_ones = [0] * N
    if S[0] == '1':
        left_ones[0] = 1
    for i in range(1, N):
        if S[i] == '1':
            left_ones[i] = left_ones[i - 1] + 1

    # `right_twos[i]` stores the number of consecutive '2's starting at index i.
    # This is calculated with a single pass from right to left.
    right_twos = [0] * N
    if S[N - 1] == '2':
        right_twos[N - 1] = 1
    for i in range(N - 2, -1, -1):
        if S[i] == '2':
            right_twos[i] = right_twos[i + 1] + 1

    # --- Step 2: Main Calculation ---
    
    max_length = 0
    # The problem guarantees S has at least one '/', so max_length will become >= 1.

    # Iterate through the string, considering each '/' as a potential center
    # of an 11/22 string.
    for i in range(N):
        if S[i] == '/':
            # Get the number of consecutive '1's immediately to the left.
            count_ones = left_ones[i - 1] if i > 0 else 0
            
            # Get the number of consecutive '2's immediately to the right.
            count_twos = right_twos[i + 1] if i < N - 1 else 0
            
            # An 11/22 string requires k '1's and k '2's. The maximum
            # possible k is limited by the minimum of available counts.
            k = min(count_ones, count_twos)
            
            # The length of the resulting 11/22 string is 2*k + 1.
            current_length = 2 * k + 1
            
            # Update the overall maximum length found so far.
            max_length = max(max_length, current_length)

    print(max_length)

if __name__ == "__main__":
    main()