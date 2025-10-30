import sys
from collections import Counter

def main():
    S = sys.stdin.readline().strip()
    # Condition 1: Length must be even
    if len(S) % 2 != 0:
        print("No")
        return

    # Condition 2: Each adjacent pair must consist of the same character
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            print("No")
            return

    # Condition 3: Each character that appears must appear exactly twice
    counts = Counter(S)
    for cnt in counts.values():
        if cnt != 2:
            print("No")
            return

    print("Yes")

if __name__ == "__main__":
    main()