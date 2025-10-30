def main():
    import sys
    
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    # Both strings are guaranteed to be the same length
    n = len(S)
    
    # The set of characters that can be replaced by '@'
    magic = {'a', 't', 'c', 'o', 'd', 'e', 'r'}
    
    # Count frequencies of letters in S and T, ignoring '@'
    from collections import Counter
    
    countS = Counter()
    countT = Counter()
    
    # Also count how many '@' in each string
    atS = 0
    atT = 0
    
    for ch in S:
        if ch == '@':
            atS += 1
        else:
            countS[ch] += 1
            
    for ch in T:
        if ch == '@':
            atT += 1
        else:
            countT[ch] += 1
    
    # Check letters that are not in magic
    # They must match exactly because we cannot turn '@' into them
    all_chars = set(countS.keys()) | set(countT.keys())
    for ch in all_chars:
        if ch not in magic:
            if countS[ch] != countT[ch]:
                print("No")
                return
    
    # For letters in magic, we can use '@' from the other string to fix mismatches.
    # We define how many letters T has in surplus for each magic letter, i.e. countT[ch] - countS[ch] if positive.
    # S must have enough '@' to compensate that surplus. And vice versa.
    
    surplus_for_S = 0  # This accumulates the total number of letters T has more than S among magic letters
    surplus_for_T = 0  # This accumulates the total number of letters S has more than T among magic letters
    
    for ch in magic:
        diff = countT[ch] - countS[ch]
        if diff > 0:
            surplus_for_S += diff
        else:
            surplus_for_T += (-diff)  # negative diff means S has more
    
    # Check if we have enough '@' in each side
    if surplus_for_S <= atS and surplus_for_T <= atT:
        print("Yes")
    else:
        print("No")

# Do not forget to call main
main()