import bisect
from collections import defaultdict
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+M]))
    
    positions = defaultdict(list)
    for idx, val in enumerate(A):
        positions[val].append(idx)
    
    # Check earliest
    earliest = []
    current = -1
    for i in range(M):
        b = B[i]
        if b not in positions:
            print("No")
            return
        lst = positions[b]
        idx_ = bisect.bisect_right(lst, current)
        if idx_ == len(lst):
            print("No")
            return
        current = lst[idx_]
        earliest.append(current)
    
    # Check latest
    latest = []
    current = N  # N is larger than any index (A is 0-based)
    for i in reversed(range(M)):
        b = B[i]
        if b not in positions:
            print("No")
            return
        lst = positions[b]
        # Find largest index in lst <= current
        idx_ = bisect.bisect_right(lst, current) - 1
        if idx_ < 0:
            print("No")
            return
        current = lst[idx_]
        latest.append(current)
    latest.reverse()
    
    if earliest != latest:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()