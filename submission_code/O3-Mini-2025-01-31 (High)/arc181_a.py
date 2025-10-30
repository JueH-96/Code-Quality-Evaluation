def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    t = int(data[0])
    index = 1
    out_lines = []

    # For each test case...
    for _ in range(t):
        if index >= len(data): break
        n = int(data[index]); index += 1
        # Read the permutation; note that P is a permutation of 1..n.
        P = list(map(int, data[index:index+n]))
        index += n

        # Case 0: Already sorted?
        already = True
        for i, p in enumerate(P):
            if p != i+1:
                already = False
                break
        if already:
            out_lines.append("0")
            continue

        # Precompute prefix minimums and maximums.
        prefix_min = [0] * n
        prefix_max = [0] * n
        prefix_min[0] = P[0]
        prefix_max[0] = P[0]
        for i in range(1, n):
            # For the prefix from index 0 to i:
            prefix_min[i] = P[i] if P[i] < prefix_min[i-1] else prefix_min[i-1]
            prefix_max[i] = P[i] if P[i] > prefix_max[i-1] else prefix_max[i-1]
                
        # Precompute suffix minimums and maximums.
        suffix_min = [0] * n
        suffix_max = [0] * n
        suffix_min[n-1] = P[n-1]
        suffix_max[n-1] = P[n-1]
        for i in range(n-2, -1, -1):
            suffix_min[i] = P[i] if P[i] < suffix_min[i+1] else suffix_min[i+1]
            suffix_max[i] = P[i] if P[i] > suffix_max[i+1] else suffix_max[i+1]

        # Check if one operation can do it.
        # (We are looking for an index i (0-indexed) so that P[i] == i+1 and
        #  the elements to the left are exactly 1..i, and those to the right are i+2..n.)
        op1_possible = False
        for i in range(n):
            if P[i] != i+1:
                continue
            # If the pivot is at an end, then automatically the parts are “correct”
            if i == 0 or i == n-1:
                op1_possible = True
                break
            # Check left part [0, i-1] must equal {1,2,..., i} – so min = 1 and max = i.
            if prefix_min[i-1] == 1 and prefix_max[i-1] == i:
                # Check right part [i+1, n-1] must equal {i+2, i+3, …, n} – so min = i+2 and max = n.
                if suffix_min[i+1] == i+2 and suffix_max[i+1] == n:
                    op1_possible = True
                    break

        if op1_possible:
            out_lines.append("1")
        else:
            # If no 1–move solution then check the extra "bad" case.
            if P[0] == n and P[-1] == 1:
                out_lines.append("3")
            else:
                out_lines.append("2")
    sys.stdout.write("
".join(out_lines))
    
if __name__ == "__main__":
    main()