import sys

def main():
    """
    Reads input, solves the problem, and prints the result.
    """
    try:
        # Read the number of mochi, N.
        N = int(sys.stdin.readline())
        
        # Read the sorted list of mochi sizes.
        A = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # Handle potential empty input during local testing.
        return

    # Initialize the total count of possible kagamimochi.
    total_count = 0
    
    # 'i' is a pointer that counts the number of valid top mochi for a given bottom mochi.
    # 'j' is the index iterating through all possible bottom mochi.
    i = 0
    for j in range(N):
        # For the current bottom mochi A[j], advance the 'i' pointer.
        # A mochi A[k] is a valid top if 2 * A[k] <= A[j].
        # Since A is sorted, as j increases, A[j] is non-decreasing.
        # Thus, the pointer 'i' only needs to move forward.
        while i < N and 2 * A[i] <= A[j]:
            i += 1
        
        # For the bottom mochi A[j], all mochi from A[0] to A[i-1] are valid tops.
        # The number of such mochi is exactly 'i'.
        # The condition that the top and bottom mochi must be distinct is
        # automatically satisfied, as proven in the explanation.
        total_count += i
        
    # Print the final result.
    print(total_count)

if __name__ == "__main__":
    main()