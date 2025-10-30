import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())

    # Read N strings
    strings = []
    for _ in range(N):
        strings.append(sys.stdin.readline().strip())

    # Sort the strings by length in ascending order
    # The problem guarantees distinct lengths, so no need for tie-breaking
    sorted_strings = sorted(strings, key=len)

    # Concatenate the sorted strings
    result = "".join(sorted_strings)

    # Print the result
    print(result)

if __name__ == "__main__":
    solve()