def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast pointer for reading
    ptr = 0
    
    # Fenwick (Binary Indexed) Tree for maintaining sums of A.
    # Allows O(log N) update and O(log N) prefix-sum query.
    class Fenwick:
        def __init__(self, n):
            self.n = n
            self.fw = [0]*(n+1)
        
        def build(self, arr):
            for i, val in enumerate(arr):
                self.add(i, val)
        
        def add(self, idx, val):
            i = idx+1
            while i <= self.n:
                self.fw[i] += val
                i += i & -i
        
        def sum(self, idx):
            # sum from 0..idx
            s = 0
            i = idx+1
            while i>0:
                s += self.fw[i]
                i -= i & -i
            return s
        
        def range_sum(self, l, r):
            if r<l: return 0
            return self.sum(r) - self.sum(l-1) if l>0 else self.sum(r)
        
        def update(self, idx, newval):
            # newval here is the delta = new_value - old_value
            self.add(idx, newval)
    
    # read N
    N = int(sys.argv[0] if False else sys.stdin.readline())  # just for safety; real code uses input_data
    # but we already have data in input_data, so:
    N = int(input_data[ptr]); ptr+=1
    
    # read A array
    A = list(map(int, input_data[ptr:ptr+N]))
    ptr += N
    
    # read B array
    B = list(map(int, input_data[ptr:ptr+N]))
    ptr += N
    
    # Build Fenwick tree for A
    fenwA = Fenwick(N)
    fenwA.build(A)
    
    # read Q
    Q = int(input_data[ptr]); ptr+=1
    
    out = []
    
    # Helper to do the greedy step:
    # We do "v = v + A_i" if v*(B_i - 1) < A_i;
    # otherwise we do "v = v * B_i".
    # But we treat B_i=1 as a special case (forced addition).
    
    for _ in range(Q):
        t = int(input_data[ptr]); ptr+=1
        
        if t==1:
            # query "1 i x" => A[i-1] = x
            i = int(input_data[ptr]); ptr+=1
            x = int(input_data[ptr]); ptr+=1
            i -= 1  # to 0-based
            # Update Fenwick tree
            delta = x - A[i]
            A[i] = x
            fenwA.update(i, delta)
        
        elif t==2:
            # query "2 i x" => B[i-1] = x
            i = int(input_data[ptr]); ptr+=1
            x = int(input_data[ptr]); ptr+=1
            i -= 1  # to 0-based
            B[i] = x
        
        else:
            # query "3 l r" => compute the maximum final value
            l = int(input_data[ptr]); ptr+=1
            r = int(input_data[ptr]); ptr+=1
            l -= 1
            r -= 1
            
            v = 0
            idx = l
            # We'll scan from idx to r in order.
            # We "bulk-add" for consecutive B=1, otherwise do the threshold check one by one.
            while idx <= r:
                if B[idx] == 1:
                    # find a maximal run of B=1
                    start = idx
                    while idx <= r and B[idx] == 1:
                        idx += 1
                    # add all A in [start..idx-1]
                    s = fenwA.range_sum(start, idx-1)
                    v += s
                else:
                    # threshold check: if v*(B[idx]-1) < A[idx], do addition, else multiply
                    # we do it for exactly one step
                    # B[idx] > 1
                    if v*(B[idx]-1) < A[idx]:
                        v += A[idx]
                    else:
                        v = v * B[idx]
                    idx += 1
            
            out.append(str(v))
    
    print("
".join(out))

# Do not forget to call main()!
if __name__ == "__main__":
    main()