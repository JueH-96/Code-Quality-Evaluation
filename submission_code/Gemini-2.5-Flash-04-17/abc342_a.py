import sys
from collections import Counter

def solve():
    S = sys.stdin.readline().strip()

    # Count character frequencies
    counts = Counter(S)

    # Find the character that appears only once
    unique_char = None
    for char, count in counts.items():
        if count == 1:
            unique_char = char
            break

    # Find the index of the unique character in the string
    # S.index(unique_char) finds the first occurrence
    index_0based = S.index(unique_char)

    # The problem asks for a 1-based index
    index_1based = index_0based + 1

    print(index_1based)

solve()