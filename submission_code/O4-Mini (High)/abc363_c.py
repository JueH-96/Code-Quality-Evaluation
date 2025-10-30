import sys
import threading
def main():
    import sys

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    S = next(it)

    # Count letters
    counts_map = [0] * 26
    for ch in S:
        counts_map[ord(ch) - 97] += 1
    # Build counts_list for distinct letters
    counts_list = [c for c in counts_map if c > 0]
    M = len(counts_list)
    # Precompute factorials up to N
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i-1] * i
    # Total permutations of S
    total_perm = fact[N]
    for c in counts_list:
        total_perm //= fact[c]
    # Quick check: if no palindrome of length K is possible
    # We need floor(K/2) pairs to form a palindrome
    need_pairs = K // 2
    sum_pairs = 0
    for c in counts_list:
        sum_pairs += (c // 2)
    if sum_pairs < need_pairs:
        # No palindrome of length K can be formed
        # All permutations avoid it
        sys.stdout.write(str(total_perm))
        return

    # Prepare palindrome-pair constraints for each window
    p = N - K + 1  # number of start positions
    # Precompute pairs for each window w: positions to union for palindrome of length K
    half = K // 2
    pairs_list = []
    for w in range(p):
        # For j in 0..half-1, pair (w+j, w+K-1-j)
        lst = [(w + j, w + K - 1 - j) for j in range(half)]
        pairs_list.append(lst)

    # DSU utilities
    # We'll declare find/union inside mask loop for speed

    # Zeros tuple for final dp state
    zeros = tuple(0 for _ in range(M))
    # The initial counts tuple
    init_counts = tuple(counts_list)
    max_count = max(counts_list)

    ans = total_perm  # include mask=0 term with sign +1

    # Loop over non-empty subsets of windows
    # Use inclusion-exclusion: subtract mask of odd size, add mask of even size
    for mask in range(1, 1 << p):
        # Determine sign
        # sign = +1 if even popcount, -1 if odd
        if (mask.bit_count() & 1) == 1:
            sign = -1
        else:
            sign = 1

        # Build DSU
        parent = list(range(N))
        def find(x):
            # Path compression
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        # Apply palindrome constraints unions
        m = mask
        w = 0
        while m:
            if (m & 1) != 0:
                # apply window w
                for (u, v) in pairs_list[w]:
                    union(u, v)
            m >>= 1
            w += 1

        # Compute cluster sizes
        cnt = [0] * N
        for i in range(N):
            r = find(i)
            cnt[r] += 1
        clusters = [sz for sz in cnt if sz > 0]

        # If any cluster size > max_count, impossible to assign
        # no letter can cover that cluster
        if max(clusters) > max_count:
            continue

        # For mask==0 we would use total_perm, but mask>=1 here
        # We do DP to count assignments of clusters to letter counts
        # Process larger clusters first to prune quickly
        clusters.sort(reverse=True)

        dp = {init_counts: 1}
        # DP over clusters
        for s in clusters:
            new_dp = {}
            # For each partial remaining-counts state
            for rem_counts, ways in dp.items():
                # Try assign this cluster of size s to each letter i
                # that has at least s remaining
                # Represent rem_counts as tuple
                # Convert to list only once per letter change
                for i in range(M):
                    if rem_counts[i] >= s:
                        # Subtract s from rem_counts[i]
                        # Build new tuple
                        # Note: build list and tuple for key
                        nc = list(rem_counts)
                        nc[i] -= s
                        nt = tuple(nc)
                        new_dp[nt] = new_dp.get(nt, 0) + ways
            dp = new_dp
            if not dp:
                break

        # Number of ways for full assignment is dp[zeros]
        cnt_mask = dp.get(zeros, 0)
        if cnt_mask:
            ans += sign * cnt_mask

    # Output the result
    sys.stdout.write(str(ans))


if __name__ == "__main__":
    main()