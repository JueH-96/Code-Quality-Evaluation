# YOUR CODE HERE
import sys

def solve():
    W, B = map(int, sys.stdin.readline().split())

    # The base string representing one period of the infinite piano keyboard pattern.
    # It has 7 'w's and 5 'b's, total length 12.
    base_string = "wbwbwwbwbwbw"

    # Determine a sufficiently long test string.
    # The maximum sum of W+B is 100+100=200.
    # The length of the base_string is 12.
    # To ensure any segment of length up to 200 (W+B_max) that might wrap around
    # the periodic boundary is covered, we need to repeat the base_string enough times.
    # A segment starting at index `i` within the infinite string, with max length `L_max`,
    # will reach index `i + L_max - 1`.
    # Due to periodicity, we only need to check starting `i` within the first period (0 to 11).
    # So the maximum index we need to cover in our test_string is `(len(base_string) - 1) + L_max - 1`.
    # (12 - 1) + 200 - 1 = 11 + 200 - 1 = 210.
    # Therefore, test_string needs to have a length of at least 211 characters (0-indexed up to 210).
    # 12 * N >= 211 => N >= 211/12 => N >= 17.58..., so N=18 repetitions are sufficient.
    test_string = base_string * 18
    
    n = len(test_string)

    # Iterate through all possible starting positions i
    for i in range(n):
        current_w_count = 0
        current_b_count = 0
        # Iterate through all possible ending positions j, starting from i
        for j in range(i, n):
            char = test_string[j]
            if char == 'w':
                current_w_count += 1
            else: # char == 'b'
                current_b_count += 1
            
            # Check if the current segment (from index i to j) matches the target W and B counts
            if current_w_count == W and current_b_count == B:
                print("Yes")
                return # Found a match, exit the function

    # If the loops complete without finding any matching segment
    print("No")

# Call the solve function to run the program
solve()