# YOUR CODE HERE
def check_conditions(S):
    # Check if the sequence is monotonically non-decreasing
    is_non_decreasing = all(S[i] <= S[i + 1] for i in range(len(S) - 1))
    
    # Check if all values are between 100 and 675 inclusive
    in_range = all(100 <= s <= 675 for s in S)
    
    # Check if all values are multiples of 25
    multiples_of_25 = all(s % 25 == 0 for s in S)
    
    # Return Yes if all conditions are satisfied, otherwise No
    return "Yes" if is_non_decreasing and in_range and multiples_of_25 else "No"

# Read input
S = list(map(int, input().strip().split()))

# Print the result
print(check_conditions(S))