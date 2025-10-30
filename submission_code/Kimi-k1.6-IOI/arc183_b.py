import bisect
import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N, K = int(input[ptr]), int(input[ptr+1])
        ptr +=2
        A = list(map(int, input[ptr:ptr+N]))
        ptr +=N
        B = list(map(int, input[ptr:ptr+N]))
        ptr +=N
        
        A_dict = defaultdict(list)
        B_dict = defaultdict(list)
        for idx, val in enumerate(A):
            A_dict[val].append(idx)
        for idx, val in enumerate(B):
            B_dict[val].append(idx)
        
        possible = True
        for i in range(N):
            b_val = B[i]
            L = max(0, i - K)
            R = min(N-1, i + K)
            found = False
            
            # Check in A's window
            if b_val in A_dict:
                indices = A_dict[b_val]
                left = bisect.bisect_left(indices, L)
                right = bisect.bisect_right(indices, R)
                if right > left:
                    found = True
            
            if found:
                continue
            
            # Check in B's window excluding current index
            if b_val in B_dict:
                indices = B_dict[b_val]
                left = bisect.bisect_left(indices, L)
                right = bisect.bisect_right(indices, R)
                count = right - left
                if count > 0:
                    if count > 1:
                        found = True
                    else:
                        idx_in_B = indices[left]
                        if idx_in_B != i:
                            found = True
            
            if not found:
                possible = False
                break
        
        print("Yes" if possible else "No")

if __name__ == "__main__":
    main()