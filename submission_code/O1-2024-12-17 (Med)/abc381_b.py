def main():
    S = input().strip()
    
    # Condition 1: |T| must be even
    if len(S) % 2 != 0:
        print("No")
        return
    
    # Condition 2: For each i, (2i-1)-th and 2i-th characters are equal
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            print("No")
            return
    
    # Condition 3: Each character appears exactly 2 times
    from collections import Counter
    char_count = Counter(S)
    for c in char_count:
        if char_count[c] != 2:
            print("No")
            return
    
    print("Yes")

# Do not remove this call
main()