def main():
    import sys
    S = sys.stdin.readline().strip()
    
    # Condition 1: length of S must be even
    if len(S) % 2 != 0:
        print("No")
        return
    
    # Condition 2: For each i, the (2i-1)-th and (2i)-th characters must be equal
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            print("No")
            return

    # Condition 3: Each character must appear exactly twice
    from collections import Counter
    freq = Counter(S)
    for ch in freq:
        if freq[ch] != 2:
            print("No")
            return
    
    print("Yes")

# Do not remove the next line
main()