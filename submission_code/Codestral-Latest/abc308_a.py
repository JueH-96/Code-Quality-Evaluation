# YOUR CODE HERE
def check_conditions(S):
    # Check if the sequence is monotonically non-decreasing
    if not all(S[i] <= S[i + 1] for i in range(len(S) - 1)):
        return "No"

    # Check if all values are between 100 and 675, inclusive
    if not all(100 <= S[i] <= 675 for i in range(len(S))):
        return "No"

    # Check if all values are multiples of 25
    if not all(S[i] % 25 == 0 for i in range(len(S))):
        return "No"

    return "Yes"

# Read input
S = list(map(int, input().split()))

# Check conditions and print result
print(check_conditions(S))