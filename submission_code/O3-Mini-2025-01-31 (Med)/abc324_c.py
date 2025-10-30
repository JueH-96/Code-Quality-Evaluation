def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return

    # Read the first line: N and T'
    header = data[0].split()
    N = int(header[0])
    t_str = header[1]
    
    # Read the next N strings S_1, S_2, ..., S_N
    strings = data[1:1+N]

    # Helper: Check if by inserting exactly one letter into 'small' we get 'large'.
    # (Assumes len(large) == len(small) + 1)
    def is_one_insertion(small, large):
        i, j = 0, 0
        used_insertion = False
        while i < len(small) and j < len(large):
            if small[i] == large[j]:
                i += 1
                j += 1
            else:
                if used_insertion:
                    return False
                used_insertion = True
                j += 1
        # If we haven't used an insertion, the extra letter is at the end.
        return True

    # For a given candidate s and received string t, check if s can be the original string T.
    # That is, t must be one of:
    # 1) s exactly (no change),
    # 2) s with one letter inserted (so len(t) == len(s)+1),
    # 3) s with one letter deleted (so len(t) == len(s)-1),
    # 4) s with one letter replaced (so len(t) == len(s) and differ in exactly one position).
    def possible_candidate(s, t):
        ls, lt = len(s), len(t)
        if ls == lt:
            # Check: either exactly equal or differ by exactly one character.
            diff = 0
            for i in range(ls):
                if s[i] != t[i]:
                    diff += 1
                    if diff > 1:
                        return False
            return diff <= 1
        elif ls + 1 == lt:
            # t is obtained by inserting one letter into s.
            return is_one_insertion(s, t)
        elif ls == lt + 1:
            # t is obtained by deleting one letter from s.
            return is_one_insertion(t, s)
        else:
            return False

    valid_indices = []
    for idx, s in enumerate(strings, start=1):
        if possible_candidate(s, t_str):
            valid_indices.append(idx)

    # Prepare output: first the count, then the list of valid indices (if any).
    out_lines = [str(len(valid_indices))]
    if valid_indices:
        out_lines.append(" ".join(map(str, valid_indices)))
    sys.stdout.write("
".join(out_lines))


if __name__ == '__main__':
    main()