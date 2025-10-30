import sys

def solve():
    S = sys.stdin.readline().strip()

    # Extract the numeric part of the string S
    # S is guaranteed to be "ABC" followed by three digits.
    # So, S[3:] gives the three-digit number string.
    contest_number_str = S[3:]
    contest_number = int(contest_number_str)

    # Define the valid ranges and specific number as per the problem description
    is_valid = False

    # Condition 1: ABC001 to ABC315
    if 1 <= contest_number <= 315:
        is_valid = True
    # Condition 2: ABC317 to ABC348 (Note: ABC316 is skipped)
    elif 317 <= contest_number <= 348:
        is_valid = True
    # Condition 3: ABC349
    elif contest_number == 349:
        is_valid = True

    # Print "Yes" if the contest number is valid, otherwise print "No"
    if is_valid:
        print("Yes")
    else:
        print("No")

# Call the solve function to run the program
if __name__ == '__main__':
    solve()