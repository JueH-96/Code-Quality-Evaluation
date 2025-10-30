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
        
        # Preprocess A's positions
        pos = defaultdict(list)
        for idx, num in enumerate(A):
            pos[num].append(idx)
        
        possible = True
        for i in range(N):
            v = B[i]
            if v not in pos:
                possible = False
                break
            lst = pos[v]
            left = i - K
            right = i + K
            # Find the first position >= left
            lo = bisect.bisect_left(lst, left)
            # Check if any element in lst up to right
            found = False
            for j in lst[lo:]:
                if j > right:
                    break
                found = True
                break
            if not found:
                possible = False
                break
        print("Yes" if possible else "No")

if __name__ == "__main__":
    main()