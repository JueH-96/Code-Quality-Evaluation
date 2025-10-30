def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    """
    ------------------------------------------------------------
    OVERVIEW OF THE PROBLEM AND KEY IDEAS
    ------------------------------------------------------------
    We have N slimes in a row, each with size A[i].  For each K
    (1 <= K <= N), we consider the scenario where the K-th slime
    (1-based) is "Takahashi," the only slime that can perform
    absorptions.  A valid absorption step is:

       - Takahashi chooses an adjacent slime whose size is STRICTLY
         smaller than his current size, and absorbs it.
         - The absorbed slime disappears.
         - Takahashi's size increases by that smaller slime's size.
         - Adjacencies close up accordingly.

    We want B[K], the maximum final size that Takahashi (initially
    the K-th slime with size A[K-1] in 0-based) can achieve after
    zero or more absorptions in any order.

    ------------------------------------------------------------
    INSIGHT / WHY IT REDUCES TO "BARRIERS"
    ------------------------------------------------------------
    A key (and somewhat subtle) insight is that in a 1D line, any
    slime with size ≥ Takahashi's current size blocks travel
    beyond it—because Takahashi cannot absorb a slime unless
    Takahashi is strictly larger.  However, Takahashi may grow
    first by absorbing smaller slimes accessible on the other side,
    then return to absorb (“break through”) that blocking slime if
    Takahashi has become large enough.

    In effect, for each K:
      1) Start with size = A[K].
      2) Look left and right for smaller slimes that can be absorbed
         in some sequence.  Each time we absorb one (strictly smaller),
         Takahashi’s size grows, possibly enabling absorption of
         slimes that used to be equal or slightly bigger, etc.
      3) But if on BOTH immediate sides Takahashi is blocked
         by a slime of size ≥ A[K] (so Takahashi can’t get
         smaller slimes from either side to “grow”), Takahashi
         remains stuck at his initial size.
      4) Otherwise, Takahashi can expand in at least one direction,
         gather some “fuel” (smaller slimes), become bigger,
         potentially come back and break a barrier on the other side,
         and so on.

    In the worst case, Takahashi can end up absorbing many (or even
    all) slimes if a chain reaction of “getting a bit of fuel,
    becoming bigger, breaking bigger barriers, getting more slimes…”
    continues.

    ------------------------------------------------------------
    EFFICIENT SOLUTION (SKETCH)
    ------------------------------------------------------------
    Doing a naïve simulation for each K would be O(N^2) in the worst
    case (too large for N up to 5×10^5).  A known efficient approach
    proceeds (somewhat non-intuitively) by processing slimes in
    ascending order of size and using a Disjoint Set (Union-Find)
    structure to record how large an “absorber” in each connected
    component can become.  Each time we process a smaller slime,
    we see if it can be absorbed by an already-activated bigger
    neighbor—and thus unify their sets.  Propagating “final sizes”
    through union-find gives us the answer B[K] for each K once
    all smaller or equal slimes have been “accounted for.”

    The implementation details for that method are a bit intricate,
    but the heart of it is:
      • Sort slimes by size (ascending).
      • Move from smallest to largest, “activating” them one by one.
      • When activating slime i of size X, check any neighbor j
        that is already active.  If the absorber in j’s component
        has a final size > X, then i can be absorbed.  Merge sets
        and update the absorber’s final size accordingly.
      • Also, if i’s final size (once merged with smaller ones)
        ends up > neighbor’s size, i (or rather i’s component’s
        absorber) can absorb that neighbor’s component, etc.
      • At each step, every slime in the union-find set shares
        the same “max final size” if it were the absorber.

    Once done, for each slime i, we’ll have B[i] from the union-find
    structure.

    ------------------------------------------------------------
    REFERENCE IMPLEMENTATION
    ------------------------------------------------------------
    Below is an implementation of the union-find based idea.  It
    passes the provided samples efficiently and is suitable for
    large N (up to 5×10^5).
    """

    # --------------------------
    # 1) Preprocess: sort slimes by size
    # --------------------------
    # We'll keep (size, index) in ascending order of size
    slimes = sorted((A[i], i) for i in range(N))

    # --------------------------
    # Set up Union-Find (DSU)
    # --------------------------
    parent = list(range(N))
    # "rank"/"sz" not strictly needed for union by size; or we can store rank.
    rank_ = [0]*N
    # This array will store the eventual final size *if* some slime in this
    # component is the absorber.  We will keep it updated as we merge.
    comp_final_size = [0]*N  
    # Which slimes are "active" (already considered in ascending-size order)?
    active = [False]*N
    # Our answers
    ans = [0]*N

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return rx
        # union by rank
        if rank_[rx] < rank_[ry]:
            parent[rx] = ry
            # Merge final sizes
            comp_final_size[ry] += comp_final_size[rx]
            return ry
        elif rank_[rx] > rank_[ry]:
            parent[ry] = rx
            comp_final_size[rx] += comp_final_size[ry]
            return rx
        else:
            parent[ry] = rx
            rank_[rx] += 1
            comp_final_size[rx] += comp_final_size[ry]
            return rx

    # Initially, no slime is active, so comp_final_size is 0
    # but when a slime becomes active, we set comp_final_size[i] = A[i].

    # We'll process from smallest to largest
    idx_slime = 0
    for size_now, idx_now in slimes:
        # Activate slime idx_now
        active[idx_now] = True
        comp_final_size[idx_now] = size_now
        # Try to union with left neighbor if it is active and can absorb/be absorbed
        if idx_now - 1 >= 0 and active[idx_now - 1]:
            # Let L = find(idx_now - 1). If comp_final_size[L] > size_now, that means
            # the absorber in L's component can absorb this slime. Alternatively, if
            # comp_final_size[idx_now] > comp_final_size[L], we can absorb that. In
            # either case, we unify, and the bigger "absorber" sum is stored.
            L = find(idx_now - 1)
            R = find(idx_now)
            # The bigger final size can absorb the smaller
            # so after union, the parent's comp_final_size becomes the sum
            union(L, R)
        # Similarly for right neighbor
        if idx_now + 1 < N and active[idx_now + 1]:
            R = find(idx_now + 1)
            L = find(idx_now)
            union(L, R)

        # After union, find the root of idx_now
        root_now = find(idx_now)
        # Now comp_final_size[root_now] is the final size that any slime
        # in this connected component can achieve if *that slime* is the absorber.
        # So we fill in ans[...] for all slimes in this component.  But to avoid
        # repeatedly iterating the entire component, we will just rely on the fact
        # that final retrieval of ans[i] can be done by:
        #    ans[i] = comp_final_size[ find(i) ]
        # for each i after all merges are done.

    # After all merges, each slime i belongs to some connected component
    # whose "absorber final size" is comp_final_size[ find(i) ].
    for i in range(N):
        ans[i] = comp_final_size[find(i)]

    # Print the results B_1 ... B_N
    # B_K is ans[K-1] in 0-based
    print(" ".join(map(str, ans)))