def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Read inputs
    N, M = map(int, input_data[:2])
    T = [0]*M
    W = [0]*M
    S = [0]*M
    idx = 2
    for i in range(M):
        T[i] = int(input_data[idx]); 
        W[i] = int(input_data[idx+1]); 
        S[i] = int(input_data[idx+2])
        idx += 3
    
    # Fenwick (Binary Indexed) Tree for "active" flags.
    # fenwicksum(x): sum of fenwicks up to x (1-based indexing).
    # fenwickupdate(x, d): add d to index x.
    # fenwicksearch(k): find smallest index i s.t. fenwicksum(i) >= k, or -1 if none.
    class FenwickTree:
        def __init__(self, n):
            self.n = n
            self.data = [0]*(n+1)
        
        def update(self, i, delta):
            while i <= self.n:
                self.data[i] += delta
                i += i & -i
        
        def query(self, i):
            s = 0
            while i > 0:
                s += self.data[i]
                i -= i & -i
            return s
        
        def total(self):
            return self.query(self.n)
        
        # find smallest i such that fenwicksum(i) >= k
        # if not found, return -1
        def search(self, k):
            if k < 1 or self.total() < k:
                return -1
            s = 0
            pos = 0
            bit_mask = 1 << (self.n.bit_length())  # a power of 2 >= n
            while bit_mask > 0:
                nxt = pos + bit_mask
                if nxt <= self.n and s + self.data[nxt] < k:
                    s += self.data[nxt]
                    pos = nxt
                bit_mask >>= 1
            return pos + 1
    
    # Build a Fenwick tree with all N people initially active (1).
    fenw = FenwickTree(N)
    for i in range(1, N+1):
        fenw.update(i, 1)
    
    # Prepare array for results
    result = [0]*N
    
    # We'll process the M noodle events in ascending order of T.
    # Also maintain a min-heap for rejoin events of form (rejoin_time, person).
    import heapq
    rejoin_heap = []
    
    # Sort the noodle events by T just in case (the problem states T_1<...<T_M, so they are already sorted).
    events = list(zip(T, W, S))  # (time, noodles, S)
    
    e = 0  # index for events
    current_time = 0
    
    while e < M:
        t_e, w_e, s_e = events[e]
        # First, process all rejoin events up to time t_e
        while rejoin_heap and rejoin_heap[0][0] <= t_e:
            r_time, r_person = heapq.heappop(rejoin_heap)
            # Mark that person as active again
            fenw.update(r_person, 1)
        
        # Now handle the noodle event at time t_e
        # Find front person if any
        if fenw.total() > 0:
            front = fenw.search(1)  # get smallest index that is active
            # front is 1-based index
            if front != -1:
                # This person gets the noodles
                result[front-1] += w_e
                # remove them from active
                fenw.update(front, -1)
                # schedule rejoin
                rejoin_time = t_e + s_e
                heapq.heappush(rejoin_heap, (rejoin_time, front))
        # else no one is active => no one gets noodles
        
        e += 1
    
    # No need to process rejoin events after the last noodle flow,
    # since we only need total noodles got, not final positions.
    
    # Output results
    print('
'.join(map(str, result)))

def main():
    solve()

if __name__ == "__main__":
    main()