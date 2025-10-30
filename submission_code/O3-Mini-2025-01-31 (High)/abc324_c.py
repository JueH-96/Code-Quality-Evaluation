def main():
    import sys
    data = sys.stdin.buffer.read().splitlines()
    if not data:
        return
    # Read first line: N and the altered string T'
    parts = data[0].split()
    N = int(parts[0])
    t = parts[1].decode() if isinstance(parts[1], bytes) else parts[1]
    lt = len(t)
    
    # We'll collect the indices (1-indexed) of candidates that could be T.
    valid_indices = []
    line_index = 1
    for idx in range(1, N+1):
        # Read candidate S_i
        s = data[line_index].decode() if isinstance(data[line_index], bytes) else data[line_index]
        line_index += 1
        ls = len(s)
        
        # The operations allowed imply that |length(S) - length(T')| <= 1.
        if abs(ls - lt) > 1:
            continue
        
        # Case 1 and 4: S and T' are of the same length.
        if ls == lt:
            # If they are exactly equal, S could be T (no change).
            if s == t:
                valid_indices.append(idx)
            else:
                # Otherwise, check if exactly one substitution can turn S to T'.
                diff = 0
                for a, b in zip(s, t):
                    if a != b:
                        diff += 1
                        if diff > 1:
                            break
                if diff == 1:
                    valid_indices.append(idx)
        
        # Case 2: T' is obtained by inserting one letter into S.
        # Then, ls + 1 == lt.
        elif ls + 1 == lt:
            i_s = 0
            i_t = 0
            used_skip = False
            valid = True
            while i_s < ls and i_t < lt:
                if s[i_s] == t[i_t]:
                    i_s += 1
                    i_t += 1
                else:
                    if not used_skip:
                        used_skip = True
                        i_t += 1
                    else:
                        valid = False
                        break
            # It is valid if all characters of s were matched.
            if valid and i_s == ls:
                valid_indices.append(idx)
        
        # Case 3: T' is obtained by deleting one letter from S.
        # Then, ls == lt + 1.
        elif ls == lt + 1:
            i_s = 0
            i_t = 0
            used_skip = False
            valid = True
            while i_s < ls and i_t < lt:
                if s[i_s] == t[i_t]:
                    i_s += 1
                    i_t += 1
                else:
                    if not used_skip:
                        used_skip = True
                        i_s += 1
                    else:
                        valid = False
                        break
            # Even if we never skipped (or skipped later), if we've matched t completely, it works.
            if valid and i_t == lt:
                valid_indices.append(idx)
    
    # Prepare output:
    output_lines = []
    output_lines.append(str(len(valid_indices)))
    if valid_indices:
        output_lines.append(" ".join(map(str, valid_indices)))
    sys.stdout.write("
".join(output_lines))

if __name__ == '__main__':
    main()