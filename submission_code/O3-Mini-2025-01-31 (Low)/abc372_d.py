def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    # reading heights; note that heights are distinct.
    H = list(map(int, data[1:]))

    # For each building j (1-indexed), we want to determine the set of indices i (with i < j)
    # for which building j is "visible" from building i. By the problem statement, j is visible
    # from i if no building between i and j (i+1, …, j−1) is taller than building j.
    #
    # A key observation is that for a fixed building j, the only potential “blocker”
    # is the closest building to the left with height greater than H[j]. Let that index be p.
    # Pay attention to the edge case: when the building p is exactly the candidate i,
    # it is not “between” i and j.
    #
    # Thus:
    #   • If there is no building to the left taller than H[j] then any i in 1...j−1 will see j.
    #   • Otherwise, let p be the nearest index to the left (p < j) with H[p] > H[j]. Then:
    #       – For i < p, building p lies between i and j and blocks j.
    #       – For i = p: p is not between p and j, so it qualifies.
    #       – For i > p (and i < j): none of the intermediate buildings (since p is the closest
    #         building taller than H[j]) are tall enough to block j.
    #
    # Therefore, for each j, the index i qualifies if and only if i lies in the interval:
    #   [start, j - 1]  where start is defined as:
    #       if there is a p with p < j and H[p] > H[j] then start = p 
    #       otherwise, start = 1.
    #
    # This means that building j contributes 1 to c_i (the count for building i) for every i 
    # such that start ≤ i < j.
    #
    # We can accumulate these contributions into an answer array using a difference array method.
    
    # Compute for each j the index p (the nearest building to the left that is taller than H[j])
    p = [0] * (n + 1)  # using 1-indexing: p[j] will store the index p for building j
    stack = []  # will store indices in increasing order that (when interpreted with H) are in decreasing order
    for j in range(1, n + 1):
        h = H[j - 1]
        # Pop from the stack until we find a building with height greater than h.
        while stack and H[stack[-1] - 1] < h:
            stack.pop()
        if stack:
            p[j] = stack[-1]
        else:
            p[j] = 0
        stack.append(j)
    
    # Use a difference array to mark that for each building j (j>=2), every building i in [start, j-1]
    # gets a contribution of +1 (i.e. j is visible from building i).
    diff = [0] * (n + 2)
    for j in range(1, n + 1):
        if j == 1:
            # For j = 1, there are no i that come before.
            continue
        start = p[j] if p[j] > 0 else 1
        # In our view, i in the interval [start, j-1] gets +1.
        if start <= j - 1:
            diff[start] += 1
            diff[j] -= 1

    # Build the answer array by taking the prefix sums of the difference array.
    ans = [0] * (n + 1)
    current = 0
    for i in range(1, n + 1):
        current += diff[i]
        ans[i] = current

    # Prepare output for buildings 1 through n.
    # c_i for i from 1 to n are in ans[1..n]
    sys.stdout.write(" ".join(str(ans[i]) for i in range(1, n + 1)))

if __name__ == '__main__':
    main()