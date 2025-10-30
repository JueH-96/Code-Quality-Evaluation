def is_valid_contest(s):
    # Check if the string starts with 'ABC' and has a length of 6
    if not (s.startswith('ABC') and len(s) == 6):
        return False

    # Extract the contest number
    contest_number = int(s[3:])

    # Check if the contest number is within the valid range
    if contest_number < 1 or contest_number > 349:
        return False

    # Check if the contest number is not 316
    if contest_number == 316:
        return False

    return True

# Read the input from stdin
s = input().strip()

# Check if the input string is a valid contest abbreviation
if is_valid_contest(s):
    print("Yes")
else:
    print("No")