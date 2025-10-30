def main():
    import sys

    S = sys.stdin.readline().strip()
    
    # Count occurrences of each letter 'a' to 'z'
    counts = [0] * 26
    for ch in S:
        counts[ord(ch) - ord('a')] += 1

    # Find the letter with the maximum frequency.
    # If there's a tie, the smaller index (earlier alphabet) wins naturally.
    max_count = -1
    best_idx = 0
    for i in range(26):
        if counts[i] > max_count:
            max_count = counts[i]
            best_idx = i

    # Convert index back to character and print
    print(chr(best_idx + ord('a')))

# Call main to execute
main()