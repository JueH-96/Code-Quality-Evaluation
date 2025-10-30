def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    S = data[0].strip()
    T = data[1].strip()
    n = len(S)
    
    # If S equals T already, we output 0 moves.
    if S == T:
        sys.stdout.write("0")
        return
    
    # We'll iteratively transform S into T.
    current = list(S)
    result = []
    
    # Since every valid move must change exactly one index where current != T,
    # and eventually all differences will be resolved, the minimal number of moves
    # is exactly the Hamming distance.
    # To get lexicographically smallest sequence (as an array), we choose at each step
    # the change that produces the lex smallest new string.
    
    while ''.join(current) != T:
        # Among indices where current[i] != T[i] choose candidate = current with that index changed.
        best_candidate = None
        best_index = -1
        for i in range(n):
            if current[i] != T[i]:
                candidate = current.copy()
                candidate[i] = T[i]
                candidate_str = ''.join(candidate)
                if best_candidate is None or candidate_str < best_candidate:
                    best_candidate = candidate_str
                    best_index = i
        # Make the chosen move.
        current[best_index] = T[best_index]
        result.append(''.join(current))
    
    # Output results: number of moves and then each resulting string.
    out_lines = [str(len(result))]
    out_lines.extend(result)
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()