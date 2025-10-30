def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    K = int(data[0])
    S = data[1]
    T = data[2]
    
    # Because K is fixed 1, we only need to check if edit distance <= 1.
    # If S == T, then it's 0 operations.
    if S == T:
        sys.stdout.write("Yes")
        return
        
    lenS, lenT = len(S), len(T)
    
    # Check for replacement: lengths are equal.
    if lenS == lenT:
        mismatches = 0
        for a, b in zip(S, T):
            if a != b:
                mismatches += 1
                if mismatches > 1:
                    break
        if mismatches == 1:
            sys.stdout.write("Yes")
            return
    
    # Check for insertion: T must be one character longer than S.
    if lenT == lenS + 1:
        # Try to see if inserting one char into S matches T.
        i, j = 0, 0
        diff_found = False
        while i < lenS and j < lenT:
            if S[i] == T[j]:
                i += 1
                j += 1
            else:
                if diff_found:  # Already have a difference.
                    break
                diff_found = True
                j += 1  # Skip this character in T, which is considered as the inserted one.
        else:
            # If we reach the end, allow remainder.
            if j < lenT or i == lenS:
                sys.stdout.write("Yes")
                return
    
    # Check for deletion: T must be one character shorter than S.
    if lenT == lenS - 1:
        # Try to see if deleting one char from S matches T.
        i, j = 0, 0
        diff_found = False
        while i < lenS and j < lenT:
            if S[i] == T[j]:
                i += 1
                j += 1
            else:
                if diff_found:
                    break
                diff_found = True
                i += 1  # skip this char in S, simulating deletion.
        else:
            # If we reached the end, it works.
            if i < lenS or j == lenT:
                sys.stdout.write("Yes")
                return
    
    sys.stdout.write("No")

if __name__ == '__main__':
    main()