import sys

def main() -> None:
    s = sys.stdin.readline().strip()
    
    # Count occurrences of each lowercase letter
    freq = [0] * 26
    for ch in s:
        freq[ord(ch) - ord('a')] += 1

    # Find maximum frequency
    max_freq = max(freq)

    # Pick the earliest (smallest alphabetical) letter with max frequency
    for i in range(26):
        if freq[i] == max_freq:
            print(chr(i + ord('a')))
            break

if __name__ == "__main__":
    main()