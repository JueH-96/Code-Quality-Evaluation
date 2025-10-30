def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    S = input_data[1]

    seen = set()
    # Iterate through the characters in S
    for index, ch in enumerate(S):
        seen.add(ch)
        # Check if all three characters are present
        if 'A' in seen and 'B' in seen and 'C' in seen:
            # Output the number of characters checked (1-indexed)
            print(index + 1)
            return

if __name__ == '__main__':
    main()