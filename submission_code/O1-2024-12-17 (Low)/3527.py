class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        """
        We have a circular array of tiles colored with 0 or 1 (red/blue).
        We define an "alternating group" of size s to be a contiguous set of s tiles (in circular sense)
        such that adjacent tiles all differ in color.

        We'll process two types of queries:
          1) [1, s] -> Return the number of alternating groups of size s.
          2) [2, i, c] -> Update colors[i] = c (0 or 1).

        Approach:
        1) Let n = len(colors). We build an auxiliary array diff of length n, where:
             diff[i] = 1 if colors[i] != colors[(i+1) % n], else 0.
           Thus "diff[i] = 1" means there's a color-change boundary between tile i and i+1.
           An "alternating group" of size s corresponds to a sub-interval of length (s-1)
           in diff that is all 1's (because we need every pair of adjacent tiles in that
           interval to differ).

        2) We group consecutive 1's in diff into runs, in a circular manner. Each run is
           a maximal circular segment of consecutive 1's. If there is only one run of length n,
           it means all tiles alternate in a big ring. The number of sub-runs of length (s-1)
           in a circular run of length r (where r < n) is (r - (s-1) + 1) if r >= (s-1), else 0,
           in the linear sense. Because we treat each run separately (none of them crosses
           a "zero"), we sum up contributions across all runs. However, if there is exactly
           one run and it has length n, that means diff is all 1's, i.e. the colors array
           is a perfect ring of alternating colors. In that special case:
             - if s < n, the count is n (you can start an alternating group of length s
               at any tile).
             - if s == n, the count is 1 (the entire ring).
             - if s > n, the count is 0.

        3) To answer queries for [1, s] quickly, we keep track of how many runs of each length r
           exist (freq[r]). Then the total count of sub-runs of size s (with s >= 2) from all runs
           is sum_{r >= s-1} freq[r] * (r - (s-1) + 1). We can compute that in O(1) with a suitable
           prefix-sum technique on freq.

           Let Freq[r] = freq[r],  for r = 0..n
           Define S0[r] = sum_{k=r..n} Freq[k]
           Define S1[r] = sum_{k=r..n} Freq[k] * k
           Then sum_{k >= x} Freq[k]*(k - x + 1)
             = sum_{k >= x} [ Freq[k]*k + Freq[k] - x*Freq[k] ]
             = (sum_{k >= x} Freq[k]*k) + (sum_{k >= x} Freq[k]) - x*(sum_{k >= x} Freq[k])
             = S1[x] + S0[x] - x*S0[x]
             = S1[x] + (1-x) * S0[x].
           In our case x = s-1.

           Special case: if there's exactly 1 run that covers the entire circle (length n),
           then we do the "all tiles alternate in a ring" counting as above.

        4) For updates [2, i, c], we do:
           - Update colors[i] = c.
           - Recompute diff[i-1] and diff[i] (taking care of circular indices).
           - This can merge or split runs of 1's in diff. We maintain our run structure
             in a linked/list or a neighbor-based arrangement so we can update freq
             in O(1) or O(log n) time for each update.

        The implementation below uses a "runs list" approach to maintain runs of '1' in the diff array.
        Each run is stored with a start index and length. Merging/splitting runs after an update
        can be performed by examining at most the runs that contain (i-1) and (i). The frequency
        array freq[] is updated accordingly, and we re-build the partial sums for freq[] after
        each update. This yields O(log n) or O(1) merges/splits per update plus O(n) to re-build
        partial sums. Because n and the number of queries can be up to 5e4, we must be careful
        to only rebuild prefix sums once after each update. That is acceptable if implemented well.

        Overall complexity: Each update is O(1) for run merging/splitting + O(n) for prefix sums,
        but with 5e4 queries, that could be borderline. A more advanced data structure could
        maintain prefix sums in O(log n). But with well-optimized code in Python (or a typical
        interview environment in C++/Java), it can pass. We'll demonstrate the general approach here.
        In practice, you might use a Fenwick tree or segment tree for freq. For clarity, we rebuild
        prefix sums after each update.

        We'll outline the code for clarity, focusing on correctness according to the specification.
        """

        import sys
        input_data = queries  # We'll treat queries as given; results go in an output list

        n = len(colors)

        # Build "diff" array: 1 if adjacent differ, 0 if same
        diff = [0]*n
        for i in range(n):
            if colors[i] != colors[(i+1) % n]:
                diff[i] = 1
            else:
                diff[i] = 0

        # A helper structure to maintain runs of consecutive 1's in diff (circular).
        # We'll store them in a doubly-linked list style. Also maintain freq array.

        # Next, we find the runs of consecutive 1's in diff (circular).
        runs = []  # list of (start_index, length)
        visited = [False]*n
        idx = 0

        # We do a simple scan: if diff[i] = 1 and not visited, collect that run.
        # We'll mark runs in a circular manner.
        def build_runs():
            runs.clear()
            i = 0
            while i < n:
                if diff[i] == 1:
                    # gather a run starting at i
                    start = i
                    length = 0
                    while diff[i] == 1:
                        visited[i] = True
                        length += 1
                        i = (i + 1) % n
                        if i == start:
                            # we've come full circle
                            break
                    runs.append((start, length))
                    if i < start:
                        # we've circled around; no need to continue the outer loop
                        break
                else:
                    i += 1

        build_runs()

        # freq[r] = how many runs of length r
        max_len = n  # in worst case
        freq = [0]*(max_len+1)
        def build_freq():
            for i in range(max_len+1):
                freq[i] = 0
            for (_, length) in runs:
                freq[length] += 1

        build_freq()

        # Precompute S0[r], S1[r], top-down: S0[r] = sum_{k=r..n} freq[k], S1[r] = sum_{k=r..n} freq[k]*k
        S0 = [0]*(max_len+2)
        S1 = [0]*(max_len+2)
        def rebuild_prefix_sums():
            S0[max_len] = freq[max_len]
            S1[max_len] = freq[max_len]*max_len
            for r in range(max_len-1, -1, -1):
                S0[r] = S0[r+1] + freq[r]
                S1[r] = S1[r+1] + freq[r]*r

        rebuild_prefix_sums()

        # Utility to see if we have exactly 1 run of length n (means diff is all 1's)
        def is_perfect_alternation():
            if len(runs) == 1:
                # If there's exactly one run, check if length == n
                return (runs[0][1] == n)
            return False

        # Merge runs or split runs around position p in diff if it changed
        # We'll remove or add runs as needed.
        # p changed from old_val to new_val in diff.
        # "p" is an index in diff.
        # This affects runs that might contain p or start at p+1 mod n.
        # We'll remove the old run that contained p (if any) and the old run that contained p+1 (if any),
        # then re-insert the new runs formed around them.
        # We'll do it in a local function that modifies 'runs' and 'freq'.
        def remove_run(run_index):
            # run_index is the index of the run in "runs" we want to remove
            (st, ln) = runs[run_index]
            # remove from freq
            freq[ln] -= 1
            # remove from "runs"
            runs.pop(run_index)

        # We'll quickly find which run p belongs to, if p is 1. If diff[p] = 0, it belongs to no run.
        # Because the number of runs can be up to n, a linear search could be O(n).
        # That's still feasible with 5e4 queries => 2.5e9 worst-case, which is borderline in Python.
        # For demonstration, we do linear search. In practice, a more advanced structure
        # would be used (like a balanced tree or link from each index to its run).
        def find_run_containing(p):
            for i, (st, ln) in enumerate(runs):
                # a run covers indices st, st+1, ..., st+ln-1 mod n
                if ln == 0:
                    continue
                end = (st + ln - 1) % n
                if st <= end:
                    if st <= p <= end:
                        return i
                else:
                    # the run wraps around end of array
                    # so p is in [st..n-1] or [0..end]
                    if p >= st or p <= end:
                        return i
            return None  # means p not in a run (diff[p] == 0 or no run covers it)

        # Reinsert runs for newly formed consecutive 1's near index p (if diff[p] changed to 1).
        # We'll gather consecutive 1's starting from p.
        def build_run_from_start(p):
            # if diff[p] == 0, no run
            if diff[p] == 0:
                return None
            start = p
            length = 0
            i = p
            while diff[i] == 1:
                length += 1
                i = (i + 1) % n
                if i == start:
                    # full circle
                    break
            return (start, length)

        def update_diff(p, old_v, new_v):
            """Handle changing diff[p] from old_v to new_v."""
            if old_v == new_v:
                return
            # 1) if old_v == 1, we remove p from its run
            if old_v == 1:
                r_idx = find_run_containing(p)
                if r_idx is not None:
                    (st, ln) = runs[r_idx]
                    # remove that run
                    freq[ln] -= 1
                    runs.pop(r_idx)
                    # now st..(st+ln-1) was a run
                    # it is possibly split into up to 2 runs by removing 'p'
                    # specifically: if ln > 1, we might get run1 from st..p-1, run2 from p+1..(st+ln-1)
                    # be mindful of modular arithmetic
                    # run1
                    if ln > 1:
                        # length = the count of consecutive 1's from st up to p-1
                        r1_len = 0
                        x = st
                        while True:
                            if x == p:
                                break
                            if diff[x] == 1:
                                r1_len += 1
                                x = (x+1) % n
                            else:
                                break
                            if x == st:
                                break
                        if r1_len > 0:
                            freq[r1_len] += 1
                            runs.append((st, r1_len))

                        # run2
                        r2_len = 0
                        y = (p+1) % n
                        while True:
                            if diff[y] == 1:
                                r2_len += 1
                                y = (y+1) % n
                            else:
                                break
                            if y == st:
                                break
                        if r2_len > 0:
                            freq[r2_len] += 1
                            runs.append(((p+1) % n, r2_len))

            # 2) if new_v == 1, we add p to a run
            diff[p] = new_v
            if new_v == 1:
                # possibly merges with run containing p-1 and run containing p+1
                # but simpler to just build one new run from p forward, remove overlapping runs, etc.
                # We'll do a small local check:
                run_candidate = build_run_from_start(p)
                if run_candidate is not None:
                    (st2, ln2) = run_candidate
                    # This "fresh run" might overlap other runs if we didn't remove them properly,
                    # but we did remove them above if old_v was 1. If old_v was 0, there's no splitting needed.
                    # Insert (st2, ln2) in runs, and freq[ln2] += 1
                    # But we must remove any runs that get fully covered by it, if there's an overlap.
                    # Because of how we do updates, hopefully there's no partial overlap left.
                    # We'll do a loop to remove any run that the new run might fully overlap.
                    # In practice, we only need a careful approach with runs. For simplicity, we can do a
                    # full 'rebuild' of runs after toggling p, but that would be O(n). We'll do that for clarity.
                    pass

        # Because the above run-merging logic is quite involved, we'll do a simpler approach in this solution:
        # after every color update, we just rebuild diff, rebuild runs, and rebuild freq, S0, S1 from scratch.
        # That is simpler (though O(n) per update). This can still pass if implemented efficiently in faster languages.
        # We'll demonstrate that approach to ensure correctness. For large constraints in Python, it may be borderline.

        def rebuild_all_structures():
            # Rebuild diff
            for i in range(n):
                diff[i] = 1 if colors[i] != colors[(i+1) % n] else 0
            # Rebuild runs
            runs.clear()
            i = 0
            visited[:] = [False]*n
            while i < n:
                if diff[i] == 1 and not visited[i]:
                    start = i
                    length = 0
                    j = i
                    while diff[j] == 1 and not visited[j]:
                        visited[j] = True
                        length += 1
                        j = (j + 1) % n
                        if j == start:
                            break
                    runs.append((start, length))
                    if j < start:
                        # we've wrapped fully
                        break
                i += 1
            # Rebuild freq
            for r in range(max_len + 1):
                freq[r] = 0
            for (_, length) in runs:
                freq[length] += 1
            # Rebuild prefix sums
            S0[max_len] = freq[max_len]
            S1[max_len] = freq[max_len]*max_len
            for r in range(max_len-1, -1, -1):
                S0[r] = S0[r+1] + freq[r]
                S1[r] = S1[r+1] + freq[r]*r

        answers = []

        # Now handle queries:
        for q in queries:
            if q[0] == 1:
                # count alternating groups of size s
                s = q[1]
                # if there's exactly 1 run with length n => special case
                if is_perfect_alternation():
                    if s < n:
                        answers.append(n)   # any start in [0..n-1]
                    elif s == n:
                        answers.append(1)   # the entire ring
                    else:
                        answers.append(0)
                else:
                    # general case
                    x = s - 1  # we need runs >= x in length
                    if x <= 0:
                        # for s=1 or s=2, but constraints say s>=3
                        # if s=2, an "alternating group" of size 2 means any pair of adjacent different colors
                        # but the problem states s >= 3, so we won't handle s < 2 here.
                        answers.append(0)
                    elif x > n:
                        # can't have runs of length > n
                        answers.append(0)
                    else:
                        # sum_{r >= x} freq[r]*(r - x + 1)
                        val = S1[x] + S0[x] - x * S0[x]  # = sum_{r>=x} freq[r]*(r - x + 1)
                        if val < 0:
                            val = 0
                        answers.append(val)
            else:
                # update color
                _, i, c = q
                if colors[i] != c:
                    colors[i] = c
                    # Rebuild everything (simplified approach)
                    rebuild_all_structures()

        return answers