import sys

def solve():
    # Read the number of strings
    N = int(sys.stdin.readline())

    # Read the N strings into a list
    strings = []
    for _ in range(N):
        strings.append(sys.stdin.readline().strip())

    # Sort the list of strings based on their length
    # The key=len argument tells sorted() to use the length of each string for comparison
    sorted_strings = sorted(strings, key=len)

    # Concatenate the sorted strings
    concatenated_string = "".join(sorted_strings)

    # Print the result
    print(concatenated_string)

if __name__ == "__main__":
    solve()