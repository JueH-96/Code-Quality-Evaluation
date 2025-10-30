def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Parse the input tokens:
    # data[0] : N (length of string S)
    # data[1] : S (the string)
    # data[2] : Q (number of operations)
    # then 2*Q tokens for Q operations.
    it = iter(data)
    N = int(next(it))
    S = next(it)
    Q = int(next(it))
    ops = []
    for _ in range(Q):
        c = next(it)
        d = next(it)
        ops.append((c, d))
    
    # Instead of processing each operation over the full string S,
    # we compute the overall transformation on each character.
    # Let mapping[i] indicate the final letter (represented as its index)
    # to which the letter with index i (0 for 'a', 1 for 'b', etc.) is transformed.
    # Initially, each letter maps to itself.
    mapping = list(range(26))
    
    # Process the operations in reverse order.
    # For a forward operation (c, d): replace each c with d,
    # the overall transformation f (as applied to S) is the composition
    # f = f_Q  ∘ f_{Q-1} ∘ ... ∘ f_1.
    # It is standard to compute the inverse transformation by going backwards:
    #   for each operation (c,d) from last to first, set mapping[c] = mapping[d].
    for c, d in reversed(ops):
        ci = ord(c) - ord('a')
        di = ord(d) - ord('a')
        mapping[ci] = mapping[di]
    
    # Now apply the computed transformation to S.
    base = ord('a')
    res_chars = []
    for ch in S:
        orig = ord(ch) - base
        # mapping[orig] gives the final letter index; convert it back to a character.
        res_chars.append(chr(mapping[orig] + base))
    
    sys.stdout.write("".join(res_chars))
    
if __name__ == '__main__':
    main()