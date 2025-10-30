import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr +=1
    for _ in range(T):
        N, K = int(input[ptr]), int(input[ptr+1])
        ptr +=2
        A = list(map(int, input[ptr:ptr+N]))
        ptr +=N
        B = list(map(int, input[ptr:ptr+N]))
        ptr +=N
        
        pos = defaultdict(list)
        for idx, num in enumerate(A):
            pos[num].append(idx)
        
        possible = True
        for i in range(N):
            b = B[i]
            if b not in pos:
                possible = False
                break
            lst = pos[b]
            low = i - K
            high = i + K
            left = bisect.bisect_left(lst, low)
            if left < len(lst) and lst[left] <= high:
                continue
            else:
                possible = False
                break
        print("Yes" if possible else "No")

if __name__ == "__main__":
    main()