def main():
    import sys
    from collections import Counter

    data = sys.stdin.read().splitlines()
    if not data:
        return

    S = data[0].strip()
    T = data[1].strip()

    # The letters that can be substituted using '@'
    valid = set("atcoder")

    # Count frequency of characters in both strings
    countS = Counter(S)
    countT = Counter(T)

    # For any letter that is NOT in the valid substitutable set,
    # it can never be formed from an '@'. Therefore, its count must match in S and T.
    for letter in set(S+T):
        if letter == '@':
            continue
        if letter not in valid:
            if countS[letter] != countT[letter]:
                print("No")
                return

    # For letters that are in the valid set, we can try to make up the differences
    # using the '@' cards in the opposite row.
    # If S has extra occurrences of a letter x relative to T, then T must provide x using
    # its '@' cards. Similarly, if T has more of x, then S must supply that letter from its '@'.
    need_T = 0  # Additional letters needed in T via substitution from '@'
    need_S = 0  # Additional letters needed in S
    for letter in valid:
        diff = countS[letter] - countT[letter]
        if diff > 0:
            # T must supply diff copies of letter by replacing '@'
            need_T += diff
        elif diff < 0:
            # S must supply -diff copies of letter by replacing '@'
            need_S += -diff

    if countT['@'] < need_T or countS['@'] < need_S:
        print("No")
    else:
        print("Yes")
        
if __name__ == '__main__':
    main()