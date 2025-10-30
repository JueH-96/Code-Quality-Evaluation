s = input().strip()

if len(s) % 2 != 0:
    print("No")
else:
    is_valid = True
    for i in range(0, len(s), 2):
        if s[i] != s[i+1]:
            is_valid = False
            break
    if is_valid:
        from collections import Counter
        counts = Counter(s)
        for char, count in counts.items():
            if count != 2:
                is_valid = False
                break
    print("Yes" if is_valid else "No")