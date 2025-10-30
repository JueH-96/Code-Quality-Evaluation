def main():
    import sys
    data = sys.stdin.read().strip().split()
    # First line: N, T'
    N = int(data[0])
    Tprime = data[1]
    S_list = data[2:]
    
    # We define a helper function to check if Tprime could be obtained from s
    # by exactly one of the operations stated or by doing no operation at all.
    def could_be_original(s, t):
        ns, nt = len(s), len(t)
        # Case 1 + Case 4: same length => either exactly the same or differ by 1 char
        if ns == nt:
            mismatch = 0
            for i in range(ns):
                if s[i] != t[i]:
                    mismatch += 1
                    if mismatch > 1:
                        return False
            # mismatch is 0 or 1 => valid
            return mismatch <= 1
        
        # Case 3: s is exactly 1 character longer than t => check if by removing 1 char from s we get t
        elif ns == nt + 1:
            i, j = 0, 0
            used_removal = False
            while i < ns and j < nt:
                if s[i] != t[j]:
                    if used_removal:
                        return False
                    used_removal = True
                    i += 1
                else:
                    i += 1
                    j += 1
            # After the loop, check leftover
            # If we never used removal, we must remove exactly one leftover char from s
            if not used_removal:
                # We must have exactly 1 leftover in s
                return (i == ns - 1) and (j == nt)
            else:
                # We used the removal somewhere in the middle
                # So after finishing matching t, let's see if there's leftover in s
                # That leftover must be zero for it to match (since we used the removal already).
                return (j == nt) and (i == ns)
        
        # Case 2: s is exactly 1 character shorter than t => check if by inserting 1 char into s we get t
        elif ns + 1 == nt:
            i, j = 0, 0
            used_insertion = False
            while i < ns and j < nt:
                if s[i] != t[j]:
                    if used_insertion:
                        return False
                    used_insertion = True
                    j += 1
                else:
                    i += 1
                    j += 1
            if not used_insertion:
                # Then we haven't used the insertion yet => must be exactly 1 leftover in t
                return (j == nt - 1) and (i == ns)
            else:
                # We used the insertion somewhere in the middle
                # We must have consumed all s before t, and leftover in t must be zero
                return (i == ns) and (j == nt)
        
        # Otherwise, can't match
        return False
    
    ans = []
    for i, s in enumerate(S_list, start=1):
        if could_be_original(s, Tprime):
            ans.append(i)
    
    print(len(ans))
    if ans:
        print(" ".join(map(str, ans)))


# Don't forget to call main()!
if __name__ == "__main__":
    main()