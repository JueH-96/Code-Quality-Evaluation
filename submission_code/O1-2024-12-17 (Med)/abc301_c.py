def main():
    import sys
    from collections import Counter
    
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    # Set of letters to which '@' can be transformed
    atcoder_set = set("atcoder")
    
    # Count characters in S and T
    cs = Counter(S)
    ct = Counter(T)
    
    # Count how many '@' each string has
    atS = cs['@']
    atT = ct['@']
    
    # neededS: how many letters S needs from its '@'
    # neededT: how many letters T needs from its '@'
    neededS = 0
    neededT = 0
    
    for ch in "abcdefghijklmnopqrstuvwxyz":
        if ch not in atcoder_set:
            # If the letter is not in 'atcoder', counts must match exactly
            if cs[ch] != ct[ch]:
                print("No")
                return
        else:
            # If the letter is in 'atcoder', mismatches can be compensated by '@'
            diff = cs[ch] - ct[ch]
            if diff < 0:
                neededS += -diff  # S needs these many letters from '@'
            else:
                neededT += diff   # T needs these many letters from '@'
    
    # Check if we have enough '@' to compensate for mismatches
    if neededS <= atS and neededT <= atT:
        print("Yes")
    else:
        print("No")

# Call main to execute solution
if __name__ == "__main__":
    main()