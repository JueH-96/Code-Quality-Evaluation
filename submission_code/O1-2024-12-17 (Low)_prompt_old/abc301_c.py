def solve():
    import sys
    
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    # Set of letters that can be replaced by '@'
    atcoder_letters = set("atcoder")
    
    # Count frequency of each character (excluding '@') in S and T
    from collections import Counter
    count_s = Counter(S)
    count_t = Counter(T)
    
    # Number of '@' in S and T
    s_at = count_s['@']
    t_at = count_t['@']
    
    # Remove '@' entries so we don't confuse them with other letters
    del count_s['@']
    del count_t['@']
    
    # 1) Check letters not in 'atcoder' - they must match exactly
    #    because we can't change them to anything else.
    for ch in set(count_s.keys()).union(set(count_t.keys())):
        if ch not in atcoder_letters:
            if count_s[ch] != count_t[ch]:
                print("No")
                return
    
    # 2) For letters in 'atcoder', calculate the shortfall.
    #    shortfall_s: how many letters T needs from S's '@'
    #    shortfall_t: how many letters S needs from T's '@'
    shortfall_s = 0
    shortfall_t = 0
    
    for ch in atcoder_letters:
        s_count = count_s[ch]
        t_count = count_t[ch]
        
        if t_count > s_count:
            shortfall_s += (t_count - s_count)
        elif s_count > t_count:
            shortfall_t += (s_count - t_count)
    
    # 3) Check if we can fill all shortfall with available '@'
    if shortfall_s <= s_at and shortfall_t <= t_at:
        print("Yes")
    else:
        print("No")

def main():
    solve()

if __name__ == "__main__":
    main()