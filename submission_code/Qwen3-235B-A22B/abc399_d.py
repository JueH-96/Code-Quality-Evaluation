import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        A = list(map(int, input[ptr:ptr+2*N]))
        ptr += 2*N
        
        pos = dict()
        for i in range(2*N):
            a = A[i]
            if a not in pos:
                pos[a] = []
            pos[a].append(i)
        
        pair_count = defaultdict(int)
        
        for x in range(1, N+1):
            positions = pos.get(x, [])
            if len(positions) != 2:
                continue  # invalid case, should not happen per problem statement
            p, q = sorted(positions)
            if q == p + 1:
                continue  # adjacent, skip
            
            # Compute blocks_p
            blocks_p = []
            if p >= 1:
                blocks_p.append(p - 1)
            if p < (2 * N - 1):
                blocks_p.append(p)
            
            # Compute blocks_q
            blocks_q = []
            if q >= 1:
                blocks_q.append(q - 1)
            if q < (2 * N - 1):
                blocks_q.append(q)
            
            # Iterate all combinations
            for block_a in blocks_p:
                for block_b in blocks_q:
                    if block_a == block_b:
                        continue
                    if block_a < block_b:
                        key = (block_a, block_b)
                    else:
                        key = (block_b, block_a)
                    pair_count[key] += 1
        
        # Sum all c choose 2
        ans = 0
        for cnt in pair_count.values():
            ans += cnt * (cnt - 1) // 2
        
        print(ans)
        # Clear the defaultdict for next test case
        pair_count.clear()

if __name__ == "__main__":
    main()