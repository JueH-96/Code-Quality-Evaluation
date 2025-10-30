def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    S = data[0].rstrip('
')
    T = data[1].rstrip('
') if len(data) > 1 else ""
    n = len(S)
    
    # If S already equals T then no operations are needed.
    if S == T:
        sys.stdout.write("0")
        return

    # In each operation, we only change one character that is different from T.
    # Since each such change “fixes” one mismatch, the minimal number of operations is
    # the Hamming distance between S and T.
    #
    # However, we have a freedom in the order of change. Our goal is not only to use the
    # minimal number of steps but also to have the resulting sequence X = (X1, X2, ..., XM)
    # be lexicographically smallest when treated as an array.
    #
    # To achieve this we proceed greedily. Let "curr" be the current string.
    # Among all indices i where curr[i] != T[i], consider the candidate string obtained by
    # changing curr[i] to T[i]. That candidate is:
    #     candidate = curr[:i] + T[i] + curr[i+1:]
    # We then choose the lexicographically smallest candidate string. This choice makes
    # the first element of our array as small as possible; and then we apply the same idea
    # recursively to choose subsequent moves.
    
    ans = []
    curr = S
    while curr != T:
        best_candidate = None
        # For every position where the character is not yet T[i],
        # try “fixing” that letter.
        for i in range(n):
            if curr[i] != T[i]:
                candidate = curr[:i] + T[i] + curr[i+1:]
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate
        curr = best_candidate
        ans.append(curr)
    
    # Output: first line is the number of operations, followed by each intermediate string.
    out_lines = [str(len(ans))] + ans
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()