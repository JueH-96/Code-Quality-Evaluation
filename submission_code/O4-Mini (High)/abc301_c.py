def main():
    import sys
    from collections import Counter

    data = sys.stdin.read().split()
    if not data:
        return
    S, T = data[0], data[1]
    cS = Counter(S)
    cT = Counter(T)

    # The only letters we can produce from '@'
    Q = set('atcoder')

    # For any letter not in Q, the counts must already match
    for i in range(26):
        ch = chr(ord('a') + i)
        if ch not in Q and cS.get(ch, 0) != cT.get(ch, 0):
            print("No")
            return

    # Compute how many replacements each side must use
    # needS = how many @ in S must become Q-letters to match T
    # needT = how many @ in T must become Q-letters to match S
    needS = 0
    needT = 0
    for ch in Q:
        diff = cT.get(ch, 0) - cS.get(ch, 0)
        if diff > 0:
            # S is short of ch by diff
            needS += diff
        else:
            # T is short of ch by -diff
            needT += -diff

    # Check if each side has enough '@'
    if needS <= cS.get('@', 0) and needT <= cT.get('@', 0):
        print("Yes")
    else:
        print("No")

# Call main() as required
main()