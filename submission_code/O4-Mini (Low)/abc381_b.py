def main():
    import sys
    
    S = sys.stdin.readline().strip()
    
    # Condition 1: Length must be even
    n = len(S)
    if n % 2 != 0:
        print("No")
        return
    
    # Condition 2: Every adjacent pair (1-based (2i-1,2i)) must be equal
    # i.e. S[0] == S[1], S[2] == S[3], ...
    for i in range(0, n, 2):
        if S[i] != S[i+1]:
            print("No")
            return
    
    # Condition 3: Each character that appears must appear exactly twice
    from collections import Counter
    freq = Counter(S)
    for count in freq.values():
        if count != 2:
            print("No")
            return
    
    # If all conditions pass
    print("Yes")

if __name__ == "__main__":
    main()