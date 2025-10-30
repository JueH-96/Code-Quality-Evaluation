import sys

def main():
    # Read the number of strings
    N = int(sys.stdin.readline())

    # Read the N strings
    strings = []
    for _ in range(N):
        strings.append(sys.stdin.readline().strip())

    # Sort the strings based on their length
    # The problem guarantees distinct lengths, so the sort order is unique
    sorted_strings = sorted(strings, key=len)

    # Concatenate the sorted strings
    concatenated_string = "".join(sorted_strings)

    # Print the result
    print(concatenated_string)

if __name__ == "__main__":
    main()