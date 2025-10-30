class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        """
        We have a circular array of tiles, each tile is colored red (0) or blue (1).
        We define an array diff[] of length n (the same as colors), where:
            diff[i] = 1 if colors[i] != colors[(i+1) % n],
            diff[i] = 0 otherwise.
        
        A "size-k alternating group" in the circle corresponds exactly to a contiguous
        block of k tiles (in circular sense) where consecutive tiles differ in color.
        In terms of diff[], that means for a start index i, the subarray diff[i..i+(k-2)]
        (indices mod n) must all be 1 (if k >= 2; the problem statement ensures k >= 3).
        So effectively, we want sub-substrings of diff[] of length (k-1) that are all 1.
        
        Counting sub-substrings of a binary array of length (k-1) that are all 1 can be
        done by finding runs of consecutive 1's in diff[]. If we have a run of length L
        (in the circular sense), then that run contains max(0, L - (k-1) + 1) such
        sub-substrings of length (k-1).

        However, the circle and color-updates complicate matters. When we change a single
        color in colors[], up to two bits in diff[] may flip (the bits representing
        adjacency to that tile). We need to update the runs of 1's in diff[]. Then queries
        of type 1 ask for the number of size-k alternating groups, which is the sum over
        all runs of 1 of (L - (k-1) + 1) if L >= (k-1).

        To handle up to 5e4 queries efficiently:
         - We maintain diff[] in a circular manner.
         - We maintain a "run structure" of consecutive 1's in diff[], stored as intervals.
         - We also maintain a Fenwick Tree (Binary Indexed Tree) or Segment Tree keyed by
           run-length, to quickly sum up how many runs have length >= X, and the sum of those
           lengths, so that we can compute the answer to queries of type 1 in O(log n) time.
         - Each color update can cause up to two flips in diff[], which can split/merge runs
           in O(log n) time.

        Below is an implementation of this idea.
        """

        import sys
        sys.setrecursionlimit(10**7)

        n = len(colors)

        # Special case: if n==0 or trivially small (but per constraints n>=4).
        if n < 4:
            # No valid queries can come, but just guard anyway
            return []

        # Build the initial diff array of length n, circular
        diff = [0]*n
        for i in range(n):
            if colors[i] != colors[(i+1) % n]:
                diff[i] = 1
            else:
                diff[i] = 0

        # --------------------------------------------------------------------
        # Fenwicks/BIT for run lengths. The max run length in diff[] is n.
        # We'll store how many runs of length L exist (freqFenwicks)
        # and the sum of L * freq(L) for each L (sumFenwicks).
        # Then for a query "count of sub-substrings of length (k-1) all=1",
        # we set X = (k-1). The count is sum_{runs of length L >= X} [L - X + 1].
        # = sum_{L >= X} (L - X + 1)
        # = sum_{L >= X} L + sum_{L >= X} 1 - X * sum_{L >= X} 1
        # = (sum of L over L>=X) + (count of L>=X) - X*(count of L>=X).
        #
        # We'll compute:
        #   countOfLgeX = count runs with length >= X
        #   sumOfLgeX   = sum of run lengths L for L >= X
        # So the answer = sumOfLgeX + countOfLgeX - X * countOfLgeX
        #
        # We'll store freqFenwicks[L] = how many runs of length L,
        # sumFenwicks[L] = L * freqFenwicks[L].
        # The Fenwicks are built for indices up to n.
        # We'll do 1-based indexing in Fenwicks for convenience.
        # --------------------------------------------------------------------

        class Fenwick:
            def __init__(self, size):
                self.size = size
                self.fw = [0]*(size+1)
            def update(self, idx, delta):
                while idx <= self.size:
                    self.fw[idx] += delta
                    idx += idx & -idx
            def query(self, idx):
                s = 0
                while idx > 0:
                    s += self.fw[idx]
                    idx -= idx & -idx
                return s
            def range_query(self, left, right):
                if left > right:
                    return 0
                return self.query(right) - self.query(left-1)

        maxLen = n
        freqFenwicks = Fenwick(maxLen)  # counts freq of run lengths
        sumFenwicks = Fenwick(maxLen)   # counts sum of L*freq(L)

        def fenwicks_add_length(L, delta):
            # L, delta => we add 'delta' to the count of runs of length L
            # and add 'delta * L' to sumFenwicks
            if L <= 0:
                return
            freqFenwicks.update(L, delta)
            sumFenwicks.update(L, L*delta)

        def fenwicks_count_ge(X):
            # how many runs have length >= X
            if X <= 1:
                return freqFenwicks.query(maxLen)  # all runs
            if X > maxLen:
                return 0
            # count of length >= X = total freq - freq of < X
            return freqFenwicks.query(maxLen) - freqFenwicks.query(X-1)

        def fenwicks_sum_ge(X):
            # sum of lengths of runs that are >= X
            if X <= 1:
                return sumFenwicks.query(maxLen)
            if X > maxLen:
                return 0
            return sumFenwicks.query(maxLen) - sumFenwicks.query(X-1)

        # --------------------------------------------------------------------
        # Next, we store the runs of 1 in a circular sense. We'll do the following:
        # 1) Find all runs of consecutive 1's in diff[], in a circular scan.
        # 2) For each run, we store (startIndex, length, prevRunId, nextRunId).
        # 3) We'll keep an array runIdOf[i] that tells which run i is in, or -1 if diff[i] == 0.
        #
        # Then each run's length L is recorded in Fenwicks. When a bit flips in diff[],
        # we either remove a '1' from a run (splitting or shrinking) or add a '1' to form/extend runs.
        # We'll do merges/splits carefully and keep Fenwicks in sync.
        # --------------------------------------------------------------------

        runIdOf = [-1]*n  # which run does index i belong to, or -1 if diff[i]==0

        # We'll store run data in arrays, keyed by run_id (an integer).
        # For a run r:
        #   runStart[r], runLen[r], runPrev[r], runNext[r]
        # We'll store them in a dictionary or list, and keep a free-list for deleted runs.
        runStart = []
        runLenArr = []
        runPrev = []
        runNext = []
        freeIDs = []
        # current number of runs stored
        # each run has an ID from 0.. up
        def new_run(start, length):
            # create a new run
            if freeIDs:
                r = freeIDs.pop()
                runStart[r] = start
                runLenArr[r] = length
                runPrev[r] = r
                runNext[r] = r
            else:
                r = len(runStart)
                runStart.append(start)
                runLenArr.append(length)
                runPrev.append(r)
                runNext.append(r)
            # add to fenwicks
            fenwicks_add_length(length, +1)
            return r

        def delete_run(r):
            # remove from fenwicks
            L = runLenArr[r]
            fenwicks_add_length(L, -1)
            # mark freed
            runStart[r] = -1
            runLenArr[r] = 0
            runPrev[r] = r
            runNext[r] = r
            freeIDs.append(r)

        def set_neighbors(r, p, nx):
            runPrev[r] = p
            runNext[r] = nx

        def link_runs(leftR, rightR):
            # link leftR->rightR, rightR->leftR in circular sense
            runNext[leftR] = rightR
            runPrev[rightR] = leftR

        def merge_runs(r1, r2):
            # merges run r2 into r1, they must be adjacent in the circle
            # remove r2, update length. Suppose r1 end and r2 start are consecutive in the circle.
            L1 = runLenArr[r1]
            L2 = runLenArr[r2]
            # remove old lengths from fenwicks
            fenwicks_add_length(L1, -1)
            fenwicks_add_length(L2, -1)
            newL = L1 + L2
            runLenArr[r1] = newL
            fenwicks_add_length(newL, +1)
            # we also need to fix runIdOf for the bits belonging to r2
            # r2 starts at runStart[r2] for L2 bits in circular sense
            # starting from s2 = runStart[r2], for L2 bits we set runIdOf[s2..] = r1
            s2 = runStart[r2]
            for i in range(L2):
                idx = (s2 + i) % n
                runIdOf[idx] = r1
            delete_run(r2)
            return r1

        def split_run(r, splitPos):
            # split run r at index splitPos into two runs [rPart1, rPart2],
            # where splitPos is in [start..start+length-1].
            # We'll assume splitPos is the exact index in the circle.
            # The part2 starts at splitPos, so part1 ends right before splitPos.
            start0 = runStart[r]
            L0 = runLenArr[r]
            # remove old length
            fenwicks_add_length(L0, -1)

            # how many bits from start0 to splitPos-1?
            offset = (splitPos - start0) % n
            if offset < 0:
                offset += n
            part1Len = offset
            part2Len = L0 - offset
            # part1 remains in run r, part2 is a new run
            runLenArr[r] = part1Len
            fenwicks_add_length(part1Len, +1)
            # reassign runIdOf for part2
            newR = new_run(splitPos, part2Len)

            # fix runIdOf
            for i in range(part2Len):
                idx = (splitPos + i) % n
                runIdOf[idx] = newR

            # link r and newR in the circular sense
            rightOfR = runNext[r]
            link_runs(r, newR)
            link_runs(newR, rightOfR)
            return newR

        # --------------------------------------------------------------------
        # Build initial runs of 1 in diff[] by a circular scan.
        # --------------------------------------------------------------------
        runCount = 0
        i = 0
        while i < n:
            if diff[i] == 0:
                runIdOf[i] = -1
                i += 1
                continue
            # we found a 1-run
            startIdx = i
            length = 0
            while diff[i] == 1:
                runIdOf[i] = -1  # will set properly after we create the run
                length += 1
                i += 1
                if i == n:
                    break
            # create the run
            r = new_run(startIdx, length)
            # assign runIdOf
            for j in range(length):
                idx = (startIdx + j) % n
                runIdOf[idx] = r

        # Now link all runs in a doubly circular list in the order they appear (by ascending runStart).
        # We'll gather all run IDs from 0..(some max). We'll skip deleted ones (start==-1).
        runIDs = [rid for rid in range(len(runStart)) if runStart[rid] != -1]
        # sort by runStart[rid]
        runIDs.sort(key=lambda x: runStart[x] if runStart[x]>=0 else n+1)

        for i in range(len(runIDs)):
            r = runIDs[i]
            rnext = runIDs[(i+1) % len(runIDs)] if len(runIDs)>0 else r
            link_runs(r, rnext)

        # --------------------------------------------------------------------
        # Helper to get the run ID of an index i in diff[]. If diff[i]==0, returns -1.
        # --------------------------------------------------------------------
        def get_run_id(i):
            return runIdOf[i]

        # --------------------------------------------------------------------
        # Flip a bit in diff[] at position pos (0->1 or 1->0),
        # and update the runs data structure accordingly.
        # --------------------------------------------------------------------
        def flip_diff_bit(pos):
            oldVal = diff[pos]
            newVal = 1 - oldVal
            diff[pos] = newVal
            if newVal == 1:
                # 0->1. We are adding a '1' at pos.
                # Possibly join with run to the left (pos-1) if diff[pos-1] == 1,
                # and run to the right (pos+1) if diff[pos+1] == 1.
                leftPos = (pos - 1) % n
                rightPos = (pos + 1) % n
                leftRun = get_run_id(leftPos)
                rightRun = get_run_id(rightPos)
                if leftRun == -1 and rightRun == -1:
                    # no adjacent runs, create a new run of length 1
                    rnew = new_run(pos, 1)
                    runIdOf[pos] = rnew
                    # link it with itself (isolated)
                    link_runs(rnew, rnew)
                elif leftRun != -1 and rightRun == -1:
                    # extend leftRun by 1
                    # remove old length from fenwicks
                    L = runLenArr[leftRun]
                    fenwicks_add_length(L, -1)
                    newL = L+1
                    runLenArr[leftRun] = newL
                    fenwicks_add_length(newL, +1)
                    # reassign runIdOf for the new position
                    # new position is exactly the next index after the run's boundary
                    # We'll find the start of leftRun, which is runStart[leftRun],
                    # and that covers L bits. We'll interpret the new bit as extending at
                    # the "end" in circular sense. However, we must figure out if pos is
                    # exactly (start + L) mod n or something. The simpler approach is to
                    # see if the run can "prepend" or "append." But for correctness in a circle,
                    # we can do this: the run covers runStart[leftRun]..some range. Extending
                    # one bit means the new coverage is L+1. The start stays the same.
                    # We'll set runIdOf[pos] = leftRun.
                    runIdOf[pos] = leftRun
                elif leftRun == -1 and rightRun != -1:
                    # extend rightRun by 1
                    # but that effectively means the runStart of rightRun should shift left by 1
                    # we'll remove old length from fenwicks
                    L = runLenArr[rightRun]
                    fenwicks_add_length(L, -1)
                    newL = L+1
                    runLenArr[rightRun] = newL
                    fenwicks_add_length(newL, +1)
                    # shift the start to pos
                    runStart[rightRun] = pos
                    runIdOf[pos] = rightRun
                else:
                    # leftRun != -1 and rightRun != -1
                    # we are bridging two runs, so we'll merge them
                    if leftRun == rightRun:
                        # That means pos is inside the same run (can happen if run covers entire array).
                        # Then effectively the run is already covering everything. The array might be all 1's.
                        # Increase length from n-1 to n if it wasn't already n. But let's do a check:
                        L = runLenArr[leftRun]
                        if L < n:
                            # remove old L
                            fenwicks_add_length(L, -1)
                            newL = L+1
                            runLenArr[leftRun] = newL
                            fenwicks_add_length(newL, +1)
                            runIdOf[pos] = leftRun
                        else:
                            # run covers entire array already
                            diff[pos] = 1  # no structural change needed
                            runIdOf[pos] = leftRun
                    else:
                        # join two distinct runs into one
                        # total length = runLenArr[leftRun] + runLenArr[rightRun] + 1
                        L1 = runLenArr[leftRun]
                        L2 = runLenArr[rightRun]
                        # remove old lengths
                        fenwicks_add_length(L1, -1)
                        fenwicks_add_length(L2, -1)
                        # remove the second run from structure
                        delete_run(rightRun)
                        newL = L1 + L2 + 1
                        runLenArr[leftRun] = newL
                        fenwicks_add_length(newL, +1)
                        # reassign all bits of rightRun to leftRun
                        # the rightRun covers runStart[rightRun].. for L2 bits
                        s2 = runStart[rightRun]
                        for i2 in range(L2):
                            idx2 = (s2 + i2) % n
                            runIdOf[idx2] = leftRun
                        # also update runIdOf[pos]
                        runIdOf[pos] = leftRun
            else:
                # 1->0. We remove '1' at pos from its run (splitting or shrinking).
                r = get_run_id(pos)
                if r == -1:
                    # Already 0? Shouldn't happen if oldVal==1, but just guard
                    return
                L = runLenArr[r]
                if L == 1:
                    # The run disappears entirely
                    delete_run(r)
                    runIdOf[pos] = -1
                else:
                    # we split the run into possibly two runs (removing a single point).
                    # remove old L
                    fenwicks_add_length(L, -1)
                    # figure out the old start
                    s = runStart[r]
                    # how far is pos from s in circular sense
                    offset = (pos - s) % n
                    if offset < 0:
                        offset += n
                    # we have two segments: [s.. pos-1], [pos+1 .. s+L-1], each mod n
                    # part1 length = offset
                    # part2 length = L - offset - 1
                    part1Len = offset
                    part2Len = L - offset - 1
                    if part1Len == 0:
                        # then the new run is only part2
                        # so the runStart should shift to pos+1
                        newStart = (pos+1) % n
                        runStart[r] = newStart
                        runLenArr[r] = part2Len
                        fenwicks_add_length(part2Len, +1)
                        # reassign runIdOf for that segment
                        for i2 in range(part2Len):
                            runIdOf[(newStart + i2) % n] = r
                    elif part2Len == 0:
                        # only part1 remains in r
                        runLenArr[r] = part1Len
                        fenwicks_add_length(part1Len, +1)
                        # reassign runIdOf if the start changes or remains
                        # the start doesn't change, it's s
                        # but we just fix runIdOf for that region
                        for i2 in range(part1Len):
                            runIdOf[(s + i2) % n] = r
                    else:
                        # we split into two runs
                        # r keeps part1, a new run holds part2
                        runLenArr[r] = part1Len
                        fenwicks_add_length(part1Len, +1)
                        # reassign runIdOf for part1 region
                        for i2 in range(part1Len):
                            runIdOf[(s + i2) % n] = r
                        # make new run for part2
                        s2 = (pos+1) % n
                        r2 = new_run(s2, part2Len)
                        # reassign runIdOf for part2 region
                        for i2 in range(part2Len):
                            runIdOf[(s2 + i2) % n] = r2
                        # now link r2 to next of r, and fix r's next
                        rn = runNext[r]
                        link_runs(r, r2)
                        link_runs(r2, rn)
                # finally set runIdOf[pos] = -1
                runIdOf[pos] = -1

        # --------------------------------------------------------------------
        # Now we can process queries. For query type 1 (size K):
        # we want the number of sub-substrings of length (K-1) that are all 1 in diff[].
        # This is sum_{ run length L >= (K-1) } of (L - (K-1) + 1).
        # = sum_{L >= X} (L - X + 1), where X = K-1.
        # = sum_{L >= X} L + count_{L>=X} - X*count_{L>=X}.
        # We'll do it using Fenwicks in O(log n).
        #
        # For query type 2 (index, color), we'll update colors[], then flip the appropriate
        # bits in diff[] (up to 2 bits).
        # --------------------------------------------------------------------

        ans = []
        for q in queries:
            if q[0] == 1:
                # query: count the number of sub-substrings of length k-1 that are all 1
                k = q[1]
                X = k-1
                if X <= 0:
                    # should not happen because k>=3 => X>=2
                    ans.append(0)
                    continue
                # count runs L >= X
                cge = fenwicks_count_ge(X)
                sge = fenwicks_sum_ge(X)
                # result = sum_{L >= X} (L - X + 1)
                #        = sge + cge - X*cge
                res = sge + cge - X*cge
                ans.append(res if res > 0 else 0)
            else:
                # query: change colors[idx] to color
                idx, newColor = q[1], q[2]
                oldColor = colors[idx]
                if oldColor == newColor:
                    # no change
                    continue
                # set the color
                colors[idx] = newColor
                # flip up to 2 bits in diff
                # diff[idx-1] depends on colors[idx-1] != colors[idx]
                pos1 = (idx - 1) % n
                oldVal1 = diff[pos1]
                newVal1 = 1 if colors[pos1] != colors[idx] else 0
                if oldVal1 != newVal1:
                    flip_diff_bit(pos1)

                # diff[idx] depends on colors[idx] != colors[idx+1]
                pos2 = idx
                oldVal2 = diff[pos2]
                newVal2 = 1 if colors[idx] != colors[(idx+1) % n] else 0
                if oldVal2 != newVal2:
                    flip_diff_bit(pos2)

        return ans