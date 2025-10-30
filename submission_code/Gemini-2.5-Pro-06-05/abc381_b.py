# YOUR CODE HERE
import sys
from collections import Counter

def main():
    """
    Reads a string from stdin, checks if it's a 1122 string,
    and prints "Yes" or "No" to stdout.
    """
    S = sys.stdin.readline().strip()
    n = len(S)

    # Condition 1: The length of the string must be even.
    # The problem constraints state 1 <= |S| <= 100, so n is at least 1.
    if n % 2 != 0:
        print("No")
        return

    # Condition 2: For each pair of characters (2i-1, 2i), they must be equal.
    # In 0-based indexing, this means S[0]==S[1], S[2]==S[3], etc.
    for i in range(0, n, 2):
        if S[i] != S[i + 1]:
            print("No")
            return

    # Condition 3: Each character that appears in the string must appear exactly twice.
    # We use collections.Counter to get the frequency of each character.
    char_counts = Counter(S)
    
    # Then, we check if all frequency counts are equal to 2.
    for count in char_counts.values():
        if count != 2:
            print("No")
            return
            
    # If the program has not returned yet, all conditions have been met.
    print("Yes")

if __name__ == "__main__":
    main()