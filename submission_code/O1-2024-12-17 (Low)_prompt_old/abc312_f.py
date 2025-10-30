def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    TX = input_data[2:]
    
    pulltabs = []  # T=0
    regulars = []  # T=1
    openers = []   # T=2
    
    idx = 0
    for _ in range(N):
        T_i = int(TX[idx]); X_i = int(TX[idx+1])
        idx += 2
        if T_i == 0:
            pulltabs.append(X_i)
        elif T_i == 1:
            regulars.append(X_i)
        else:
            openers.append(X_i)
    
    # Sort descending
    pulltabs.sort(reverse=True)
    regulars.sort(reverse=True)
    openers.sort(reverse=True)
    
    # Prefix sums for pulltabs (happiness), regulars (happiness), openers (capacity)
    # prefixA[i] = sum of top i pulltabs
    # prefixB[i] = sum of top i regulars
    # prefixC[i] = total capacity from top i openers
    def build_prefix(arr):
        psum = [0]*(len(arr)+1)
        for i in range(len(arr)):
            psum[i+1] = psum[i] + arr[i]
        return psum
    
    prefixA = build_prefix(pulltabs)
    prefixB = build_prefix(regulars)
    prefixC = build_prefix(openers)
    
    import bisect
    
    max_happiness = 0
    
    # For r in [0.. up to min(M, len(regulars))], 
    # find the minimum x s.t. prefixC[x] >= r (i.e. capacity >= r),
    # then we can pick x openers + r regulars, leaving M-x-r items for pulltabs.
    # p = min(M - x - r, len(pulltabs)), total = prefixB[r] + prefixA[p].
    # keep track of the maximum such total.
    
    max_r = min(M, len(regulars))
    
    for r in range(max_r+1):
        # capacity needed = r
        # find x s.t. prefixC[x] >= r
        # do a binary search for the smallest x
        # prefixC is non-decreasing, so we want x = leftmost index with prefixC[x]>=r
        lo = 0
        hi = len(prefixC) - 1  # prefixC has length of len(openers)+1
        if prefixC[hi] < r:
            # even using all openers we can't open r cans
            continue
        
        # binary search
        # we know prefixC[len(openers)] >= r from the check above
        while lo < hi:
            mid = (lo + hi) // 2
            if prefixC[mid] >= r:
                hi = mid
            else:
                lo = mid + 1
        x = lo  # smallest index s.t. prefixC[x] >= r
        
        # now check if we can pick x openers + r cans within M
        if x + r > M:
            # not enough remaining picks
            continue
        
        # number of pulltabs we can pick
        p = M - x - r
        if p > len(pulltabs):
            p = len(pulltabs)
        
        candidate = prefixB[r] + prefixA[p]
        if candidate > max_happiness:
            max_happiness = candidate
    
    print(max_happiness)

def main():
    solve()

# For the testing environment, we'll just call solve().
if __name__ == "__main__":
    solve()