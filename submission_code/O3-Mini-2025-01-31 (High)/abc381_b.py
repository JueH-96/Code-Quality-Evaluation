def main():
    import sys
    S = sys.stdin.read().strip()
    
    # Condition 1: length should be even.
    if len(S) % 2 != 0:
        print("No")
        return

    # Condition 2: each consecutive pair of characters must be equal.
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            print("No")
            return

    # Condition 3: each character present in S must appear exactly twice.
    freq = {}
    for char in S:
        freq[char] = freq.get(char, 0) + 1
        
    for count in freq.values():
        if count != 2:
            print("No")
            return

    print("Yes")

if __name__ == '__main__':
    main()