import collections
import sys

def solve():
    N = int(sys.stdin.readline())
    
    # Read the list of integers A_i
    # sys.stdin.readline() reads the line including the newline.
    # .split() handles splitting by whitespace and effectively ignores the trailing newline.
    A_line = sys.stdin.readline()
    A = list(map(int, A_line.split()))

    # Count frequencies of each integer
    counts = collections.Counter(A)

    max_found_value = -1  # Stores the maximum integer found so far that satisfies the condition
                          # Initialized to -1, as all A_i are >= 1.
    ans_label = -1        # Stores the label of the person with max_found_value
                          # Default to -1 if no such person is found.

    # Iterate through each person
    for i in range(N):
        person_value = A[i]
        person_label = i + 1  # Person labels are 1-indexed

        # Check if this person's integer is unique
        if counts[person_value] == 1:
            # If unique, check if this integer is greater than the max found so far
            if person_value > max_found_value:
                max_found_value = person_value
                ans_label = person_label
    
    # Print the result
    sys.stdout.write(str(ans_label) + "
")

if __name__ == '__main__':
    solve()