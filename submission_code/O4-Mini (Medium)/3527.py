from typing import List
import bisect

class BIT:
    def __init__(self, n):
        self.n = n
        self.bit = [0] * (n+1)
    def add(self, i, v):
        # i: 1-based
        while i <= self.n:
            self.bit[i] += v
            i += i & -i
    def query(self, i):
        # sum[1..i]
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        # alt[i] = 1 if colors[i] != colors[(i+1)%n]
        alt = [0] * n
        zero_count = 0
        for i in range(n):
            alt[i] = 1 if colors[i] != colors[(i+1) % n] else 0
            if alt[i] == 0:
                zero_count += 1
        # BITs over run lengths 1..n
        bit_freq = BIT(n)
        bit_sum = BIT(n)
        # runs: list of [start, end] for alt==1 runs in linear 0..n-1
        runs = []
        def bit_insert_run_length(L):
            # record a run length L
            bit_freq.add(L, 1)
            bit_sum.add(L, L)
        def bit_remove_run_length(L):
            bit_freq.add(L, -1)
            bit_sum.add(L, -L)
        # initialize runs
        if zero_count == 0:
            # full run
            runs.append([0, n-1])
            bit_insert_run_length(n)
        else:
            i = 0
            while i < n:
                if alt[i] == 0:
                    i += 1
                else:
                    j = i
                    while j < n and alt[j] == 1:
                        j += 1
                    runs.append([i, j-1])
                    L = j - i
                    bit_insert_run_length(L)
                    i = j
        # helper to find run index containing position x, or None
        def find_run_containing(x):
            # find first run.start > x
            # we use runs sorted by start
            i = bisect.bisect_right(runs, [x, n])
            i -= 1
            if i >= 0 and runs[i][0] <= x <= runs[i][1]:
                return i
            return None
        # helper to find run index starting at x
        def find_run_starting(x):
            i = bisect.bisect_left(runs, [x, -1])
            if i < len(runs) and runs[i][0] == x:
                return i
            return None

        answers = []
        for q in queries:
            if q[0] == 1:
                size = q[1]
                k = size - 1
                # if single full run
                if len(runs) == 1 and runs[0][1] - runs[0][0] + 1 == n:
                    # all alt bits are 1
                    answers.append(n)
                else:
                    # sum over runs L >= k of (L - k + 1)
                    # C = count runs with length >= k
                    # S = sum of lengths of runs with length >= k
                    # prefix upto k-1:
                    c_pref = bit_freq.query(k-1)
                    s_pref = bit_sum.query(k-1)
                    tot_runs = bit_freq.query(n)
                    tot_sum = bit_sum.query(n)
                    C = tot_runs - c_pref
                    S = tot_sum - s_pref
                    ans = S - (k-1) * C
                    answers.append(ans)
            else:
                # type 2 update
                _, idx, newc = q
                if colors[idx] == newc:
                    # no change
                    continue
                colors[idx] = newc
                # positions in alt to update: p2 = idx-1, then p1 = idx
                for dpos in [idx-1, idx]:
                    p = dpos % n
                    # old alt
                    old = alt[p]
                    # compute new alt
                    # alt[p] = colors[p] != colors[(p+1)%n]
                    nxt = (p+1) % n
                    new = 1 if colors[p] != colors[nxt] else 0
                    if old == new:
                        continue
                    alt[p] = new
                    # if flip 1->0: remove or split run
                    if old == 1 and new == 0:
                        # find run containing p
                        ri = find_run_containing(p)
                        if ri is None:
                            # shouldn't happen
                            continue
                        st, ed = runs[ri]
                        oldL = ed - st + 1
                        # remove old run
                        bit_remove_run_length(oldL)
                        runs.pop(ri)
                        # left segment if any
                        if p > st:
                            L1 = p - st
                            runs.insert(ri, [st, p-1])
                            bit_insert_run_length(L1)
                            ri += 1
                        # right segment if any
                        if p < ed:
                            L2 = ed - p
                            runs.insert(ri, [p+1, ed])
                            bit_insert_run_length(L2)
                        # track zero count
                        zero_count += 1
                    # if flip 0->1: create or extend/merge runs
                    else:
                        # new == 1
                        # zero_count--
                        zero_count -= 1
                        # neighbors in alt array
                        lm = (p-1) % n
                        rm = (p+1) % n
                        left_one = (alt[lm] == 1)
                        right_one = (alt[p] == 1 and alt[rm] == 1) or (alt[rm] == 1 and p == lm)  # simpler: alt[lm], alt[p]? Actually neighbors are runs on alt: left uses p-1, right uses p+1
                        # Actually for merging, left_one = alt[lm]==1, right_one = alt[p]==1? No.
                        # We inserted alt[p]=1. The run on right is if alt[p+1]==1, we check alt[p+1].
                        left_one = (alt[lm] == 1)
                        right_one = (alt[rm] == 1)
                        if not left_one and not right_one:
                            # new single run [p,p]
                            bis = bisect.bisect_left(runs, [p, -1])
                            runs.insert(bis, [p, p])
                            bit_insert_run_length(1)
                        elif left_one and not right_one:
                            # extend left run to include p
                            # find run containing lm
                            ri = find_run_containing(lm)
                            st, ed = runs[ri]
                            oldL = ed - st + 1
                            bit_remove_run_length(oldL)
                            runs[ri][1] = p
                            newL = oldL + 1
                            bit_insert_run_length(newL)
                        elif not left_one and right_one:
                            # extend right run
                            ri = find_run_containing(rm)
                            st, ed = runs[ri]
                            oldL = ed - st + 1
                            bit_remove_run_length(oldL)
                            runs[ri][0] = p
                            newL = oldL + 1
                            bit_insert_run_length(newL)
                        else:
                            # both sides: merge left and right
                            # find left run at lm, right run at rm
                            li = find_run_containing(lm)
                            ri = find_run_containing(rm)
                            # ensure li < ri for popping sequence
                            if li > ri:
                                li, ri = ri, li
                            st_l, ed_l = runs[li]
                            st_r, ed_r = runs[ri]
                            L_l = ed_l - st_l + 1
                            L_r = ed_r - st_r + 1
                            # remove both
                            bit_remove_run_length(L_l)
                            bit_remove_run_length(L_r)
                            # delete ri first then li
                            runs.pop(ri)
                            runs.pop(li)
                            # new merged [st_l, ed_r]
                            new_st, new_ed = st_l, ed_r
                            newL = (ed_r - st_l + 1)
                            bis = bisect.bisect_left(runs, [new_st, -1])
                            runs.insert(bis, [new_st, new_ed])
                            bit_insert_run_length(newL)
        return answers