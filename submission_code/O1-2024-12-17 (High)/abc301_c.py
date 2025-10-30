def main():
    import sys

    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Both strings have the same length
    # Count how many of each a-z appear (excluding '@'), and how many '@' in each
    countS = [0]*26
    countT = [0]*26
    atS = 0
    atT = 0

    for ch in S:
        if ch == '@':
            atS += 1
        else:
            countS[ord(ch) - ord('a')] += 1

    for ch in T:
        if ch == '@':
            atT += 1
        else:
            countT[ord(ch) - ord('a')] += 1

    # Letters that can come from '@'
    allowed = set('atcoder')

    for i in range(26):
        letter = chr(i + ord('a'))
        # If the letter is NOT in the allowed set, we cannot fix its mismatch with '@'
        if letter not in allowed:
            if countS[i] != countT[i]:
                print("No")
                return
        else:
            # letter can be formed using '@'
            diff = countS[i] - countT[i]
            if diff > 0:
                # T needs diff more copies of this letter
                if atT < diff:
                    print("No")
                    return
                else:
                    atT -= diff
            elif diff < 0:
                # S needs -diff more copies of this letter
                need = -diff
                if atS < need:
                    print("No")
                    return
                else:
                    atS -= need

    # If we never failed, it's possible
    print("Yes")

# Do not forget to call main()
main()