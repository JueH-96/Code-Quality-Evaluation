def can_match(s, t):
    # Get character frequencies for each string excluding @
    s_freq = {}
    t_freq = {}
    s_at = 0
    t_at = 0
    
    for c in s:
        if c == '@':
            s_at += 1
        else:
            s_freq[c] = s_freq.get(c, 0) + 1
            
    for c in t:
        if c == '@':
            t_at += 1
        else:
            t_freq[c] = t_freq.get(c, 0) + 1
    
    # Check each character
    atcoder = 'atcoder'
    
    # For each character in s that's not in t, we need an @ in t
    for c in s_freq:
        if c not in t_freq:
            t_at -= s_freq[c]
        elif s_freq[c] > t_freq[c]:
            t_at -= (s_freq[c] - t_freq[c])
            
    # For each character in t that's not in s, we need an @ in s
    for c in t_freq:
        if c not in s_freq:
            s_at -= t_freq[c]
        elif t_freq[c] > s_freq[c]:
            s_at -= (t_freq[c] - s_freq[c])
    
    return s_at >= 0 and t_at >= 0

s = input()
t = input()

print("Yes" if can_match(s, t) else "No")