import sys

def main() -> None:
    """
    Reads an unknown number of integers (one per line) from standard input,
    where the last integer is guaranteed to be 0. Prints the integers in
    reverse order, each on its own line.
    """
    # Read all whitespace-separated tokens, convert to integers
    nums = [int(token) for token in sys.stdin.read().split()]
    
    # Output in reverse order
    for value in reversed(nums):
        print(value)

if __name__ == "__main__":
    main()