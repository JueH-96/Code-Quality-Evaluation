import sys
from collections import Counter

def main():
    # Read input from stdin
    S = sys.stdin.read().strip()

    # Count the frequency of each character
    counter = Counter(S)

    # Find the maximum frequency
    max_freq = max(counter.values())

    # Find the characters with the maximum frequency
    candidates = [char for char, freq in counter.items() if freq == max_freq]

    # Sort the candidates and pick the first one (earliest in alphabetical order)
    result = sorted(candidates)[0]

    # Write the result to stdout
    print(result)

if __name__ == "__main__":
    main()