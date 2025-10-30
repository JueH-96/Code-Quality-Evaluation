def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    Tprime = input_data[1]
    S_list = input_data[2:]
    
    # We will create a helper function to check each condition.
    # Condition:
    # 1) Tprime == T
    # 2) Tprime is obtained by inserting exactly one character into T
    # 3) Tprime is obtained by deleting exactly one character from T
    # 4) Tprime is obtained by changing exactly one character of T
    
    def differ_by_one_change(s, t):
        """
        Returns True if s and t are the same length
        and they differ in exactly 0 or 1 positions (i.e., either equal or 1 char changed).
        Otherwise returns False.
        """
        if len(s) != len(t):
            return False
        
        diff_count = 0
        for c1, c2 in zip(s, t):
            if c1 != c2:
                diff_count += 1
                if diff_count > 1:
                    return False
        return diff_count <= 1  # 0 or 1 difference is allowed
    
    def differ_by_one_insertion(s, t):
        """
        Check if t can be obtained by inserting exactly one character into s.
        Equivalently, check if s can match t if t has exactly one extra character.
        So len(t) = len(s) + 1
        We check with 2-pointer technique if s can match t by skipping exactly one char in t.
        """
        if len(t) != len(s) + 1:
            return False
        
        i = 0  # pointer for s
        j = 0  # pointer for t
        used_skip = False
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                if used_skip:
                    return False
                used_skip = True
                j += 1
        # If we exit the loop, consider leftover. 
        # j might still point to the inserted char if not used.
        return True  # because we can skip at most once
    
    def differ_by_one_deletion(s, t):
        """
        Check if t can be obtained by deleting exactly one character from s.
        Equivalently, check if s can match t if s has exactly one extra character.
        So len(s) = len(t) + 1
        We check with 2-pointer technique if t can match s by skipping exactly one char in s.
        """
        if len(s) != len(t) + 1:
            return False
        
        i = 0  # pointer for s
        j = 0  # pointer for t
        used_skip = False
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                if used_skip:
                    return False
                used_skip = True
                i += 1
        return True
    
    # We now check for each S_i whether it could be T.
    
    results = []
    iidx = 1
    for s in S_list:
        # Compare lengths
        len_diff = len(Tprime) - len(s)
        if len_diff == 0:
            # either same string or differ by one change
            if differ_by_one_change(s, Tprime):
                results.append(iidx)
        elif len_diff == 1:
            # Tprime might be obtained by inserting 1 char into s
            # so Tprime is s with 1 insertion => check differ_by_one_insertion(s, Tprime)
            if differ_by_one_insertion(s, Tprime):
                results.append(iidx)
        elif len_diff == -1:
            # Tprime might be obtained by deleting 1 char from s
            # so Tprime is s with 1 deletion => check differ_by_one_deletion(s, Tprime)
            if differ_by_one_deletion(s, Tprime):
                results.append(iidx)
        # else len_diff not in [-1, 0, 1], not possible
        iidx += 1
    
    # Output results
    print(len(results))
    if results:
        print(" ".join(map(str, results)))