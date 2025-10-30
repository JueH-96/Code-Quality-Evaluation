def main():
    import sys
    S = sys.stdin.read().strip()
    
    # Check if string length is even.
    if len(S) % 2 != 0:
        print("No")
        return

    # Check that every two adjacent characters are the same.
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            print("No")
            return

    # Check that every character appears exactly twice.
    freq = {}
    for ch in S:
        freq[ch] = freq.get(ch, 0) + 1

    for count in freq.values():
        if count != 2:
            print("No")
            return

    print("Yes")

if __name__ == '__main__':
    main()