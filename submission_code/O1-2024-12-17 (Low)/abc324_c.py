def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    Tprime = input_data[1]
    S_list = input_data[2:]
    
    len_t = len(Tprime)
    
    def check_equal_or_change(s1, s2):
        # Check if s1 and s2 are either equal or differ by exactly one character
        # Assume len(s1) == len(s2)
        diff_count = 0
        for c1, c2 in zip(s1, s2):
            if c1 != c2:
                diff_count += 1
                if diff_count > 1:
                    return False
        # diff_count <= 1 means it's valid
        return diff_count <= 1
    
    def check_insertion(shorter, longer):
        # Check if longer can be obtained from shorter by inserting exactly one character
        # Assume len(longer) = len(shorter) + 1
        i, j = 0, 0
        mismatch_used = False
        while i < len(shorter) and j < len(longer):
            if shorter[i] == longer[j]:
                i += 1
                j += 1
            else:
                if mismatch_used:
                    return False
                mismatch_used = True
                j += 1
        # If we exit the loop, we can have at most one mismatch so far
        return True  # The rest can differ by at most one char (the inserted one)
    
    answers = []
    idx = 1
    
    for S in S_list:
        len_s = len(S)
        
        # Case 1: lengths are the same
        if len_s == len_t:
            # T' could be S_i if they are equal (0 mismatch) or differ by exactly 1 mismatch
            if check_equal_or_change(S, Tprime):
                answers.append(idx)
        
        # Case 2: S has length one less => T' might come from S by inserting 1 char
        elif len_s + 1 == len_t:
            # T' is from S by insertion => check if Tprime can be obtained from S by adding 1 char
            if check_insertion(S, Tprime):
                answers.append(idx)
        
        # Case 3: S has length one more => T' might come from S by deleting 1 char
        elif len_s - 1 == len_t:
            # T' is from S by deleting => check if S can be obtained from T' by insertion or vice versa
            # i.e. check if S is T' with 1 insertion => check_insertion(Tprime, S)
            if check_insertion(Tprime, S):
                answers.append(idx)
        
        idx += 1
    
    print(len(answers))
    if answers:
        print(" ".join(map(str, answers)))

# call main() to comply with the requirement
if __name__ == "__main__":
    main()