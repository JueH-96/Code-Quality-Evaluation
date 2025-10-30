def main():
    import sys
    import bisect
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = int(input_data[1])
    A = list(map(int, input_data[2:]))

    # Quick check: if S < min(A), it's impossible (all A_i >= 1).
    # Because all elements are positive, any non-empty subsequence sum >= min(A).
    if S < min(A):
        print("No")
        return

    # Compute prefix sums p: p[i] = sum of A[0..i-1], so p[0] = 0, p[N] = sum(A).
    p = [0] * (N+1)
    for i in range(N):
        p[i+1] = p[i] + A[i]
    block_sum = p[N]

    # 1) Check if there's a sub-subsequence within a single block = A (standard two-pointer).
    left = 0
    current_sum = 0
    for right in range(N):
        current_sum += A[right]
        while current_sum > S and left <= right:
            current_sum -= A[left]
            left += 1
        if current_sum == S:
            print("Yes")
            return

    # 2) If block_sum == S, we can take the entire block.
    if block_sum == S:
        print("Yes")
        return

    # 3) If S is a multiple of block_sum and at least one block, we can take k full blocks.
    if block_sum != 0 and S % block_sum == 0:
        k = S // block_sum
        if k >= 1:  # means sum of k blocks = S
            print("Yes")
            return
        # else continue

    # Now we attempt the "crossing-boundary / multiple-blocks" case.
    #
    # We want to find a non-empty subsequence that spans at least one block boundary,
    # i.e. sum = (k+1)*block_sum + prefix(r) - prefix(l)  (0 <= l < N+1, 0 <= r < N+1)
    # reindexed so that p[i] = sum of first i elements, i in [0..N].
    #
    # That sum equals S  iff  (k+1)*block_sum = S + [p[l] - p[r]].
    # Let offset(l,r) = p[l] - p[r].  Then we need S + offset(l,r) >= block_sum (to have k+1 >= 1)
    # and (S + offset(l,r)) mod block_sum == 0.
    #
    # Define T = (block_sum - (S mod block_sum)) mod block_sum.
    # Then offset(l,r) mod block_sum == T, and offset(l,r) >= block_sum - S.
    #
    # We'll build P[i] = p[i] mod block_sum, group the p[i] by P[i], and then do a search.

    # If block_sum = 0 cannot happen here since A_i >=1 => block_sum>0, but let's still guard.
    if block_sum == 0:
        print("No")
        return

    # T = (block_sum - (S mod block_sum)) % block_sum
    # (If S >= block_sum, S mod block_sum < block_sum; if S < block_sum, same formula still works.)
    M = S % block_sum
    T = (block_sum - M) % block_sum  # This is the remainder we need from offset(l,r).

    # Build array of p[i] mod block_sum
    P = [p[i] % block_sum for i in range(N+1)]

    from collections import defaultdict
    groups = defaultdict(list)
    # Group indices by their P[i] value, storing actual p[i].
    for i in range(N+1):
        groups[P[i]].append(p[i])

    # Sort each list so we can binary-search by p[i].
    for key in groups:
        groups[key].sort()

    # Now, for each l in [0..N], we want offset(l,r) = p[l] - p[r].
    # We need offset(l,r) mod block_sum = T => (p[l] - p[r]) mod block_sum = T
    # => P[l] = (P[r] + T) mod block_sum => P[r] = (P[l] - T) mod block_sum
    # Also offset(l,r) >= block_sum - S => p[l] - p[r] >= (block_sum - S)
    # => p[r] <= p[l] - (block_sum - S) = p[l] + (S - block_sum).

    import math

    for l in range(N+1):
        mod_l = P[l]
        needed_mod = (mod_l - T) % block_sum
        x = p[l] + (S - block_sum)  # We want p[r] <= x
        candidate_list = groups.get(needed_mod, [])
        # Find how many are <= x
        idx = bisect.bisect_right(candidate_list, x)
        if idx > 0:
            # That means there's at least one p[r] <= x
            print("Yes")
            return

    # If none found, print No
    print("No")

# Do not forget to call main!
if __name__ == "__main__":
    main()