# YOUR CODE HERE
import sys

class BIT:
    __slots__ = ['N', 'tree']
    def __init__(self, size):
        self.N = size
        self.tree = [0]*(self.N+2)
    
    def add(self, idx, value):
        tree = self.tree
        while idx <= self.N:
            tree[idx] += value
            idx += idx & -idx
    
    def sum(self, idx):
        res = 0
        tree = self.tree
        while idx > 0:
            res += tree[idx]
            idx -= idx & -idx
        return res

def main():
    import sys
    import sys
    input = sys.stdin.read().split()
    ptr =0
    N = int(input[ptr])
    ptr +=1
    Q = int(input[ptr])
    ptr +=1
    S_str = input[ptr]
    ptr +=1
    S = [0]*(N+2)  # 1-based indexing, S[1..N]
    for i in range(1, N+1):
        S[i] = int(S_str[i-1])
    # Initialize BITs
    BIT_flips = BIT(N)
    BIT_matching = BIT(N-1)
    for i in range(1, N):
        if S[i] == S[i+1]:
            BIT_matching.add(i,1)
    outputs = []
    for _ in range(Q):
        if ptr >= len(input):
            break
        query_type = input[ptr]
        ptr +=1
        L = int(input[ptr])
        ptr +=1
        R = int(input[ptr])
        ptr +=1
        if query_type == '1':
            # Flip operation
            # Handle L-1 and R
            if L >1:
                a = S[L-1] ^ (BIT_flips.sum(L-1) %2)
                b = S[L] ^ (BIT_flips.sum(L) %2)
                old_match = (a == b)
                new_b = b ^1
                new_match = (a == new_b)
                if old_match != new_match:
                    if new_match:
                        BIT_matching.add(L-1,1)
                    else:
                        BIT_matching.add(L-1,-1)
            if R <N:
                a = S[R] ^ (BIT_flips.sum(R) %2)
                b = S[R+1] ^ (BIT_flips.sum(R+1) %2)
                old_match = (a == b)
                new_a = a ^1
                new_match = (new_a == b)
                if old_match != new_match:
                    if new_match:
                        BIT_matching.add(R,1)
                    else:
                        BIT_matching.add(R,-1)
            # Apply flip
            BIT_flips.add(L,1)
            BIT_flips.add(R+1,-1)
        else:
            # Query operation
            if L == R:
                outputs.append("Yes")
            else:
                cnt = BIT_matching.sum(R-1) - BIT_matching.sum(L-1)
                if cnt ==0:
                    outputs.append("Yes")
                else:
                    outputs.append("No")
    print('
'.join(outputs))

if __name__ == "__main__":
    main()