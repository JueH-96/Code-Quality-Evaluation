import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    P = list(map(int, input[ptr:ptr + N]))
    ptr += N
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr + M]))
    ptr += M

    class FenwickTree():
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 1)
        
        def update(self, idx, delta):
            while idx <= self.n:
                self.tree[idx] += delta
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res

    def compute_inversion():
        ft = FenwickTree(N)
        inv_count = 0
        for num in reversed(P):
            inv_count += ft.query(num - 1)
            ft.update(num, 1)
        return inv_count
    
    initial_inv = compute_inversion()
    swap_count = 0
    current_max_k = 0
    de = deque()
    output = []

    for k in A:
        k_1based = k
        if k_1based > current_max_k:
            current_max_k = k_1based
            processed = set()
            new_de = deque()
            # Enqueue all positions from current_max_k-1 downto 1 (since k can be up to N)
            # Positions are 1-based, so new positions are from previous current_max_k+1 up to k_1based-1
            prev_max = current_max_k - 1 if current_max_k != 0 else 0
            new_start = prev_max + 1
            for i in range(new_start, current_max_k):
                new_de.append(i)
            # Merge new_de into de
            de.extend(new_de)
        
        swaps = True
        current_de = deque()
        while swaps:
            swaps = False
            while de:
                i = de.popleft()
                if i >= current_max_k - 1:
                    continue
                # Check if swap is needed at position i (0-based in de, but operations are 1-based)
                # Current code uses 1-based i here?
                # Wait, k is 1-based. So current_max_k is k, and the maximum position allowed is k-1 (1-based).
                # So positions to process are from 1 to k-1 in 1-based terms, which is 0 to k-2 in 0-based.
                # So i ranges from 0 to current_max_k - 2 (0-based)
                if i >= current_max_k - 1:
                    continue
                # Compare P[i] and P[i+1] (0-based)
                if P[i] > P[i+1]:
                    swaps = True
                    swap_count += 1
                    # Swap P[i] and P[i+1]
                    P[i], P[i+1] = P[i+1], P[i]
                    # Add adjacent positions to the deque
                    if i > 0:
                        current_de.append(i-1)
                    current_de.append(i)
                    current_de.append(i+1)
            de = current_de
            current_de = deque()
        output.append(initial_inv - swap_count)
    
    print('
'.join(map(str, output)))

if __name__ == "__main__":
    main()