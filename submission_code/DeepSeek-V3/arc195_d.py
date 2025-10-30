import sys

def solve():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    results = []
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        A = input[ptr]
        ptr += 1
        B = input[ptr]
        ptr += 1
        
        # Check feasibility
        a_ones = []
        b_ones = []
        for i in range(N):
            if A[i] == '1':
                a_ones.append(i)
        for i in range(N):
            if B[i] == '1':
                b_ones.append(i)
        
        if len(a_ones) > len(b_ones):
            results.append(-1)
            continue
        
        # Check left and right boundaries
        if len(b_ones) == 0:
            # But problem states B has at least one 1
            pass
        left_possible = True
        if len(a_ones) > 0 and len(b_ones) > 0:
            if a_ones[0] < b_ones[0]:
                left_possible = False
            if a_ones[-1] > b_ones[-1]:
                left_possible = False
        if not left_possible:
            results.append(-1)
            continue
        
        # Now find the minimal operations
        # We need to match a_ones to some subset of b_ones, left to right, with minimal max |a_i - b_j|
        # The optimal is to match a_i to b_j where j is i shifted by some delta, but in the minimal way.
        # The problem reduces to matching the a_ones to the first m b_ones (m = len(a_ones)) in a way that minimizes the maximum |a_i - b_{k + i}| for some k.
        # Or perhaps, the minimal maximum is the maximum of (b_j - a_i) for some pairs.
        # The solution is to align the k-th a_one (0-based) with the (k + delta)-th b_one, where delta is (len(b_ones) - len(a_ones)).
        # The minimal operations is the maximum of |a_i - b_{i + delta}| for i in 0..m-1.
        m = len(a_ones)
        n_b = len(b_ones)
        delta = n_b - m
        max_ops = 0
        for i in range(m):
            # The optimal is to match a_ones[i] to b_ones[i + delta_part], where delta_part is chosen to spread the remaining B's.
            # The correct approach is to match a_i to b_{i + k}, where k is (n_b - m) (so that the remaining B's are distributed).
            j = i + (n_b - m)
            max_ops = max(max_ops, abs(a_ones[i] - b_ones[j]))
        results.append(max_ops)
    
    print('
'.join(map(str, results)))

solve()