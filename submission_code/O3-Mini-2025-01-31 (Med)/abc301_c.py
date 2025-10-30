def main():
    import sys
    from collections import Counter

    data = sys.stdin.read().splitlines()
    if not data:
        return
    S = data[0].strip()
    T = data[1].strip()
    
    cnt_S = Counter(S)
    cnt_T = Counter(T)
    allowed = set("atcoder")
    
    # For each non allowed character (excluding '@'),
    # the frequency must be the same in both S and T.
    for ch in set(S + T):
        if ch == '@':  # Skip '@' because we can replace it.
            continue
        if ch not in allowed:
            if cnt_S[ch] != cnt_T[ch]:
                sys.stdout.write("No")
                return
    
    # For allowed letters, we can use the '@' as wildcards to adjust.
    # Determine how many replacements are needed in each row.
    # If S has less of an allowed letter than T needs, S must use '@' cards to cover the gap.
    # Similarly, if T has fewer allowed letters than S, T must cover that gap with '@'.
    need_S = 0  # Number of replacements needed in S.
    need_T = 0  # Number of replacements needed in T.
    
    for ch in allowed:
        if cnt_S[ch] < cnt_T[ch]:
            need_S += (cnt_T[ch] - cnt_S[ch])
        elif cnt_T[ch] < cnt_S[ch]:
            need_T += (cnt_S[ch] - cnt_T[ch])
    
    # Check if the available '@' cards in each row can cover the needed replacements.
    if cnt_S['@'] >= need_S and cnt_T['@'] >= need_T:
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")
        
if __name__ == '__main__':
    main()