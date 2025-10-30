# YOUR CODE HERE
def can_win(s, t):
    # Count frequencies of each character
    freq_s = {}
    freq_t = {}
    
    for c in s:
        freq_s[c] = freq_s.get(c, 0) + 1
    
    for c in t:
        freq_t[c] = freq_t.get(c, 0) + 1
    
    # Get the number of @ symbols in each string
    at_s = freq_s.get('@', 0)
    at_t = freq_t.get('@', 0)
    
    # Remove @ from frequency counts
    if '@' in freq_s:
        del freq_s['@']
    if '@' in freq_t:
        del freq_t['@']
    
    # Characters that @ can be replaced with
    replaceable = set(['a', 't', 'c', 'o', 'd', 'e', 'r'])
    
    # For each character, calculate the deficit
    # deficit_s: how many more of this character S needs
    # deficit_t: how many more of this character T needs
    deficit_s = 0
    deficit_t = 0
    
    all_chars = set(freq_s.keys()) | set(freq_t.keys())
    
    for char in all_chars:
        count_s = freq_s.get(char, 0)
        count_t = freq_t.get(char, 0)
        
        if count_s > count_t:
            # S has more of this character than T
            if char in replaceable:
                # T can use @ to match
                deficit_t += count_s - count_t
            else:
                # Cannot use @ for this character
                return False
        elif count_t > count_s:
            # T has more of this character than S
            if char in replaceable:
                # S can use @ to match
                deficit_s += count_t - count_s
            else:
                # Cannot use @ for this character
                return False
    
    # Check if we have enough @ symbols to cover the deficits
    return deficit_s <= at_s and deficit_t <= at_t

# Read input
s = input().strip()
t = input().strip()

# Check if it's possible to win
if can_win(s, t):
    print("Yes")
else:
    print("No")