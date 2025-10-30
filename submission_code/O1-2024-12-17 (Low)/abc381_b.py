def main():
    import sys
    S = sys.stdin.readline().strip()
    
    # Condition 1: |S| is even
    if len(S) % 2 != 0:
        print("No")
        return
    
    # Condition 2: Pairs of characters at positions (2i-1) and (2i) must match
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            print("No")
            return
    
    # Condition 3: Each character appears exactly two times
    from collections import Counter
    cnt = Counter(S)
    for char, freq in cnt.items():
        if freq != 2:
            print("No")
            return
            
    # If all conditions are satisfied, print Yes
    print("Yes")

# Do not forget to call main function
if __name__ == "__main__":
    main()