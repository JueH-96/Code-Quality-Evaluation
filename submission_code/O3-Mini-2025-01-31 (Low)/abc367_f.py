def main():
    import sys, random
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    
    # We will use two mod values and random weights to get a hash value representing the multiset of a subarray.
    # Two independent mod values minimize the probability of collisions.
    mod1 = 10**9 + 7
    mod2 = 10**9 + 9
    
    # Elements are in the range 1..n (per constraints). We generate a random weight for each possible.
    max_val = n  # since A_i, B_i <= n.
    weight1 = [0] * (max_val + 1)
    weight2 = [0] * (max_val + 1)
    random.seed(0)  # fixed seed for reproducibility
    for v in range(1, max_val + 1):
        weight1[v] = random.randint(1, mod1 - 1)
        weight2[v] = random.randint(1, mod2 - 1)
    
    # Construct prefix sums for A and B.
    # Each prefix sum represents the sum (mod mod1 and mod2 respectively) of the random weights assigned to the elements.
    prefixA1 = [0] * (n + 1)
    prefixA2 = [0] * (n + 1)
    prefixB1 = [0] * (n + 1)
    prefixB2 = [0] * (n + 1)
    for i in range(1, n + 1):
        a_val = A[i - 1]
        b_val = B[i - 1]
        prefixA1[i] = (prefixA1[i - 1] + weight1[a_val]) % mod1
        prefixA2[i] = (prefixA2[i - 1] + weight2[a_val]) % mod2
        prefixB1[i] = (prefixB1[i - 1] + weight1[b_val]) % mod1
        prefixB2[i] = (prefixB2[i - 1] + weight2[b_val]) % mod2
    
    # Process each query:
    # For a query, we want to check whether we can rearrange A[l...r] to get B[L...R].
    # This is equivalent to checking that the multisets have the same frequency counts. 
    # We compare our computed "multiset hashes" for both subarrays.
    out_lines = []
    for _ in range(q):
        l = int(next(it))
        r = int(next(it))
        L = int(next(it))
        R = int(next(it))
        # If the lengths differ, they cannot be the same multiset.
        if (r - l) != (R - L):
            out_lines.append("No")
            continue
        hashA1 = (prefixA1[r] - prefixA1[l - 1]) % mod1
        hashA2 = (prefixA2[r] - prefixA2[l - 1]) % mod2
        hashB1 = (prefixB1[R] - prefixB1[L - 1]) % mod1
        hashB2 = (prefixB2[R] - prefixB2[L - 1]) % mod2
        if hashA1 == hashB1 and hashA2 == hashB2:
            out_lines.append("Yes")
        else:
            out_lines.append("No")
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == "__main__":
    main()