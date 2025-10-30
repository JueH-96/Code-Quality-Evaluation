import sys
from collections import Counter

def main() -> None:
    S = sys.stdin.readline().strip()

    # Frequency of each letter
    letter_freq = Counter(S)

    # How many different letters occur i times
    freq_of_freq = Counter(letter_freq.values())

    max_freq = max(letter_freq.values())          # highest occurring frequency

    for i in range(1, max_freq + 1):
        if freq_of_freq.get(i, 0) not in (0, 2):
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()