import sys

def solve():
    # Read N and M from the first line of input
    # sys.stdin.readline() reads a line including the newline character, so .split() handles it
    n, m = map(int, sys.stdin.readline().split())

    # Read the elements of sequence A from the second line
    # Store them in a set for efficient O(1) average time complexity for membership testing
    a_elements = set(map(int, sys.stdin.readline().split()))

    # Initialize an empty list to store the numbers that are missing
    missing_numbers = []

    # Iterate through all integers from 1 to N (inclusive)
    for i in range(1, n + 1):
        # If the current integer 'i' is not found in our set of A_elements,
        # it means it's a missing number.
        if i not in a_elements:
            missing_numbers.append(i)

    # Output the count of missing numbers
    # len(missing_numbers) gives the count C
    sys.stdout.write(str(len(missing_numbers)) + "
")

    # Output the missing numbers themselves, space-separated
    # The numbers in 'missing_numbers' are already in ascending order because we iterated from 1 to N
    # map(str, missing_numbers) converts all integers in the list to strings
    # " ".join(...) then concatenates these strings with a space in between
    # If missing_numbers is empty, " ".join(map(str, [])) results in an empty string,
    # and sys.stdout.write("" + "
") prints just a newline, which is correct for the sample output format.
    sys.stdout.write(" ".join(map(str, missing_numbers)) + "
")

# Call the solve function to run the program
if __name__ == '__main__':
    solve()