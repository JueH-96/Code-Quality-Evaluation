import sys
from collections import Counter

def main():
    S = sys.stdin.readline().strip()

    # Condition 1: length must be even
    if len(S) % 2:
        print("No")
        return

    # Condition 2: every odd/even position pair must be equal
    for i in range(0, len(S), 2):
        if S[i] != S[i + 1]:
            print("No")
            return

    # Condition 3: every character occurring must appear exactly twice
    freq = Counter(S)
    if all(v == 2 for v in freq.values()):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()