def main():
    import sys
    from collections import Counter

    input_lines = sys.stdin.read().splitlines()
    if not input_lines:
        return
    S = input_lines[0].strip()
    T = input_lines[1].strip()

    # The set of allowed letters for replacement with '@'
    allowed = set("atcoder")

    # Count the occurrences of each character in S and T
    countS = Counter(S)
    countT = Counter(T)

    # Wildcards in each row (cards showing '@')
    s_wild = countS.get('@', 0)
    t_wild = countT.get('@', 0)

    # For letters that are NOT in the allowed set, their counts must match exactly
    # because '@' cannot be replaced with these letters.
    for ch in "abcdefghijklmnopqrstuvwxyz":
        if ch not in allowed:
            if countS.get(ch, 0) != countT.get(ch, 0):
                print("No")
                return

    # For letters in the allowed set, the differences can be compensated
    # with the wildcards. For each letter:
    #   If S fixed cards have more of a letter than T's fixed cards,
    #   then T must replace some of its '@' cards to cover the gap.
    #   Similarly, if T has more, then S must cover it.
    needS = 0  # The number of wildcards S must use to cover its deficit.
    needT = 0  # The number of wildcards T must use to cover its deficit.
    for ch in allowed:
        diff = countS.get(ch, 0) - countT.get(ch, 0)
        if diff > 0:
            # S has extra copies of ch. T must supply these via '@'
            needT += diff
        elif diff < 0:
            # T has extra copies of ch. S must supply these via '@'
            needS += -diff

    # Check if each row has enough '@' cards to fill in the deficits.
    if needS <= s_wild and needT <= t_wild:
        print("Yes")
    else:
        print("No")
        
if __name__ == '__main__':
    main()