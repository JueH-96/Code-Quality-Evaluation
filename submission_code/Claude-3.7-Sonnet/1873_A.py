def can_make_abc(s):
    # If already "abc", no swaps needed
    if s == "abc":
        return "YES"
    
    # Try all possible swaps
    for i in range(3):
        for j in range(i+1, 3):
            # Create a new string with characters swapped
            chars = list(s)
            chars[i], chars[j] = chars[j], chars[i]
            if ''.join(chars) == "abc":
                return "YES"
    
    return "NO"

t = int(input())
for _ in range(t):
    s = input().strip()
    print(can_make_abc(s))