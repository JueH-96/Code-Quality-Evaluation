import sys

# Read the integer N from standard input, representing the length of the string.
# readline() reads a line including the newline character, int() converts it to an integer.
n = int(sys.stdin.readline())

# Read the string S from standard input.
# readline() reads the line including the newline.
# strip() removes leading/trailing whitespace, including the newline character.
s = sys.stdin.readline().strip()

# Use the built-in string method find() to locate the first occurrence of the substring "ABC".
# The find() method returns the lowest index (0-based) in the string where the substring "ABC" is found.
# If the substring "ABC" is not found within the string S, find() returns -1.
index = s.find("ABC")

# Check the value returned by find().
if index == -1:
    # If index is -1, it means "ABC" was not found as a contiguous substring in S.
    # Print -1 as required by the problem statement.
    print(-1)
else:
    # If index is not -1, it holds the 0-based starting index of the first occurrence of "ABC".
    # The problem asks for the position 'n', which is 1-based.
    # To convert the 0-based index to a 1-based position, we add 1 to the index.
    # Print the calculated 1-based position.
    print(index + 1)