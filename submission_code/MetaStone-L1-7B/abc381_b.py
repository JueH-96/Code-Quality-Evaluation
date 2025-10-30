S = input().strip()

if len(S) % 2 != 0:
    print("No")
else:
    # Check each pair of consecutive characters
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            print("No")
            exit()
    
    # Check each character appears exactly twice
    counts = {}
    for c in S:
        counts[c] = counts.get(c, 0) + 1
    
    for v in counts.values():
        if v != 2:
            print("No")
            exit()
    
    print("Yes")