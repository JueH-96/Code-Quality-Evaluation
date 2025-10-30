import sys
from bisect import bisect_right, bisect_left
from collections import defaultdict

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Build the indices dictionary
    indices_dict = defaultdict(list)
    for idx, val in enumerate(A):
        indices_dict[val].append(idx)
    
    # Check if B is a subsequence and compute earliest path
    pos = []
    prev = -1
    for b in B:
        if b not in indices_dict:
            print("No")
            return
        indices = indices_dict[b]
        i = bisect_right(indices, prev)
        if i >= len(indices):
            print("No")
            return
        pos.append(indices[i])
        prev = indices[i]
    
    # Compute the latest path
    pos_late = [0] * M
    prev_late = N
    for i in range(M-1, -1, -1):
        b = B[i]
        indices = indices_dict[b]
        j = bisect_left(indices, prev_late) - 1
        if j < 0:
            print("No")
            return
        pos_late[i] = indices[j]
        prev_late = indices[j]
    
    # Compare the two paths
    for i in range(M):
        if pos[i] != pos_late[i]:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()