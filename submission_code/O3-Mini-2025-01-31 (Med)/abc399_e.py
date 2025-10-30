def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    S = data[1].strip()
    T = data[2].strip()
    
    # For every letter from 'a' to 'z', we define an initial mapping to itself.
    mapping = {chr(c): chr(c) for c in range(ord('a'), ord('z') + 1)}
    
    # The idea: for every occurrence i, if S[i] is converted eventually to some letter,
    # then we must have a consistent mapping for that letter. Otherwise, it is impossible.
    for i in range(n):
        s_char = S[i]
        t_char = T[i]
        # If we haven’t set any mapping for s_char (i.e. it is still the identity),
        # assign it. Otherwise, if it was set before, it must match.
        if mapping[s_char] == s_char:
            mapping[s_char] = t_char
        elif mapping[s_char] != t_char:
            sys.stdout.write("-1")
            return

    # Every letter x in S that gets mapped to a different letter (i.e. mapping[x] != x) must be replaced at some point.
    # Let op_count be the number of such operations initially.
    # (Even if multiple letters map to the same letter, they are done with separate operations.)
    used = set(S)
    op_count = 0
    for ch in mapping:
        if mapping[ch] != ch and ch in used:
            op_count += 1

    # However, if there is a cycle among transformations, we cannot simply replace each letter.
    # For instance, if a->b, b->c, and c->a, then we cannot do these replacements directly
    # because after one replacement, some letters are overwritten.
    # A well known trick is to break the cycle by using a temporary letter.
    # In a cycle of length k (where k >= 2), we need one extra operation.
    #
    # We now build a directed graph where an edge from x to y exists if mapping[x] != x.
    # Since each letter gets at most one target, the graph is "functional".
    # We traverse over the 26 letters and detect cycles of length > 1.
    visited = set()
    extra = 0  # extra operations needed for cycles

    letters = [chr(c) for c in range(ord('a'), ord('z') + 1)]
    for ch in letters:
        if ch in visited:
            continue
        # Use a dictionary to record the position (index) in the current exploration to detect cycles.
        rec_stack = []
        pos = {}  # letter -> index in rec_stack
        cur = ch
        while True:
            if cur in pos:
                cycle_length = len(rec_stack) - pos[cur]
                if cycle_length > 1:
                    extra += 1
                break
            pos[cur] = len(rec_stack)
            rec_stack.append(cur)
            nxt = mapping[cur]
            # If a letter maps to itself, the chain stops.
            if nxt == cur:
                break
            if nxt in visited:
                break
            cur = nxt
        for node in rec_stack:
            visited.add(node)
    
    result = op_count + extra
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    main()