from collections import defaultdict

# A simple segment tree for range maximum queries.
class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.tree = [-10**9]*(2*self.size)
        for i in range(self.n):
            self.tree[self.size+i] = data[i]
        for i in range(self.size-1, 0, -1):
            self.tree[i] = max(self.tree[2*i], self.tree[2*i+1])
    def query(self, l, r):
        # query maximum value in the interval [l, r] (inclusive)
        res = -10**9
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.tree[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.tree[r])
                r -= 1
            l //= 2
            r //= 2
        return res

# Given string s and capacity 'cap', compute for each starting index i:
#   next_arr[i] = largest index j such that s[i:j] has at most cap distinct letters;
#   Also compute r0_arr[i] = the minimal r (with r > i) such that s[i:r]
#   has exactly cap distinct letters (we store r0_arr[i] = r).
def compute_next_and_r0(s, cap):
    n = len(s)
    next_arr = [0]*(n+1)
    r0_arr = [None]*n
    freq = defaultdict(int)
    distinct = 0
    j = 0
    # Special handling: when cap==1 we want to “force” the break as soon as possible.
    for i in range(n):
        if j < i:
            j = i
            freq = defaultdict(int)
            distinct = 0
        while j < n:
            c = s[j]
            if freq[c] == 0:
                if distinct + 1 > cap:
                    break
                distinct += 1
            freq[c] += 1
            # record the first position (after i) where we have exactly cap distinct.
            if (j > i) and (r0_arr[i] is None) and (distinct == cap):
                r0_arr[i] = j+1  # we store as the partition boundary (j+1)
            j += 1
        next_arr[i] = j
        c = s[i]
        freq[c] -= 1
        if freq[c] == 0:
            distinct -= 1
    next_arr[n] = n
    return next_arr, r0_arr

# A variant of the above that only computes "next" pointer (no modification of r0).
def compute_next_only(s, cap):
    n = len(s)
    next_arr = [0]*(n+1)
    freq = defaultdict(int)
    distinct = 0
    j = 0
    for i in range(n):
        if j < i:
            j = i
            freq = defaultdict(int)
            distinct = 0
        while j < n:
            c = s[j]
            if freq[c] == 0:
                if distinct + 1 > cap:
                    break
                distinct += 1
            freq[c] += 1
            j += 1
        next_arr[i] = j
        c = s[i]
        freq[c] -= 1
        if freq[c] == 0:
            distinct -= 1
    next_arr[n] = n
    return next_arr

class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        # Compute the next pointer and r0 for capacity = k (no–modification greedy segmentation)
        next_k, r0 = compute_next_and_r0(s, k)
        # g[i] = number of partitions from s[i:] using the greedy rule (no modification)
        g = [0]*(n+1)
        g[n] = 0
        for i in range(n-1, -1, -1):
            g[i] = 1 + g[next_k[i]]
            
        # If k == 26 then the alphabet is full so you cannot force an early break.
        # In that case the best you can do is the greedy segmentation result.
        # (We will still compute the DP but then Option2 will be ignored.)
        
        # Next, for the forced state (i.e. right after modification at some index),
        # we “pretend” to have replaced a character so that the allowed capacity becomes k-1.
        # We pre–compute forcedNext for capacity = (k-1).
        if k - 1 <= 0:
            # for k==1, no extra new letter is allowed so forcedNext is trivial.
            forcedNext = [i for i in range(n+1)]
        else:
            forcedNext = compute_next_only(s, k-1)
        
        # Define a function T where for any index r (0 <= r < n) we set:
        #   T[r] = g( forcedNext[r+1] )
        # That is, if you force a break at r (by modifying s[r]) then the forced partition's remainder
        # will start from forcedNext[r+1].
        T = [0]*n
        for r in range(n):
            idx = r+1
            if idx > n:
                idx = n
            T[r] = g[ forcedNext[idx] ]
        
        # Build a segment tree to answer range maximum queries over T.
        segTree = None
        if n > 0:
            segTree = SegmentTree(T)
        
        # f[i] = maximum number of partitions obtainable from s[i:] when you still have the modification option.
        f_dp = [0]*(n+1)
        f_dp[n] = 0
        for i in range(n-1, -1, -1):
            # Option1: do not use the modification in the current partition.
            opt1 = 1 + f_dp[ next_k[i] ]
            opt2 = -10**9
            # Option2: use modification – if k < 26 (so that there is at least one letter available outside
            # the current distinct set)
            # and if we can use modification in the current partition.
            if k < 26:
                if r0[i] is not None and r0[i] < next_k[i]:
                    # For any r in [r0[i], next_k[i]), if we force a break at r
                    # we get 1 (for the current partition) + (1 + g( forcedNext[r+1] )) = 2 + g(forcedNext[r+1])
                    # We want the maximum of this value over r.
                    l = r0[i]
                    r_index = next_k[i] - 1
                    best = segTree.query(l, r_index)
                    opt2 = 2 + best
            f_dp[i] = max(opt1, opt2)
        
        # Our answer (starting at index 0 with modification available) is f_dp[0].
        return f_dp[0]

# For testing:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxPartitionsAfterOperations("accca", 2))   # expected 3
    print(sol.maxPartitionsAfterOperations("aabaab", 3))  # expected 1
    print(sol.maxPartitionsAfterOperations("xxyz", 1))    # expected 4