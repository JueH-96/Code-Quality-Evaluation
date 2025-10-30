def main():
    import sys
    S = sys.stdin.readline().strip()
    
    # Condition 1: Length must be even
    n = len(S)
    if n % 2 != 0:
        print("No")
        return
    
    # Condition 2: Characters come in adjacent equal pairs
    for i in range(0, n, 2):
        if S[i] != S[i+1]:
            print("No")
            return
    
    # Condition 3: Each character appears exactly 0 or 2 times
    from collections import Counter
    counts = Counter(S)
    for cnt in counts.values():
        if cnt != 2:
            print("No")
            return
    
    # All conditions satisfied
    print("Yes")

if __name__ == "__main__":
    main()