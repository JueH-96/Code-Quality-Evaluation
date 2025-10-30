import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    M = int(input[ptr]); ptr +=1
    Q = int(input[ptr]); ptr +=1

    pairs = [[] for _ in range(M)]
    for i in range(M):
        S = int(input[ptr]); ptr +=1
        T = int(input[ptr]); ptr +=1
        if S < T:
            a = S
            b = T-1
            p1 = (a-1, 0)
            p2 = (b-1, 1)
        else:
            a = T
            b = S-1
            p1 = (a-1, 1)
            p2 = (b-1, 0)
        pairs[i] = [p1, p2]
    
    for _ in range(Q):
        L = int(input[ptr]); ptr +=1
        R = int(input[ptr]); ptr +=1
        seen = defaultdict(int)
        conflict = False
        for i in range(L-1, R):  # i is 0-based person index
            p1 = pairs[i][0]
            p2 = pairs[i][1]
            for idx, typ in [p1, p2]:
                mask = 1 << typ
                if seen[idx] & mask:
                    continue
                seen[idx] |= mask
                if seen[idx] == 3:  # both 0 and 1
                    conflict = True
                    break
            if conflict:
                break
        print("No" if conflict else "Yes")

if __name__ == "__main__":
    main()