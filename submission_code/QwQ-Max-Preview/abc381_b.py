s = input().strip()

# Check if the length is even
if len(s) % 2 != 0:
    print("No")
else:
    # Check each consecutive pair
    valid = True
    for i in range(0, len(s), 2):
        if s[i] != s[i+1]:
            valid = False
            break
    if not valid:
        print("No")
    else:
        # Check each character appears exactly twice
        from collections import Counter
        counts = Counter(s)
        for c in counts.values():
            if c != 2:
                valid = False
                break
        print("Yes" if valid else "No")