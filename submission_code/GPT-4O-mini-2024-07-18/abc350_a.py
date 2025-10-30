def is_valid_contest_abbreviation(S):
    # Create a set of valid contest numbers
    valid_contests = {f"ABC{str(i).zfill(3)}" for i in range(1, 350) if i != 316}
    
    # Check if the input string is in the set of valid contests
    return "Yes" if S in valid_contests else "No"

# Read input
S = input().strip()

# Output the result
print(is_valid_contest_abbreviation(S))