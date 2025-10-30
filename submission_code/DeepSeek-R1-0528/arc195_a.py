import sys
import bisect
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    if not data:
        print("No")
        return
    
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+n]))
    B = list(map(int, data[2+n:2+n+m]))
    
    mapping = defaultdict(list)
    for i, a in enumerate(A):
        mapping[a].append(i)
    
    last = -1
    left_seq = []
    for b in B:
        if b not in mapping:
            print("No")
            return
        arr = mapping[b]
        pos = bisect.bisect_right(arr, last)
        if pos == len(arr):
            print("No")
            return
        index_in_A = arr[pos]
        left_seq.append(index_in_A)
        last = index_in_A
        
    last = n
    right_seq = []
    for i in range(len(B)-1, -1, -1):
        b = B[i]
        arr = mapping[b]
        pos = bisect.bisect_left(arr, last)
        if pos == 0:
            print("No")
            return
        index_in_A = arr[pos-1]
        right_seq.append(index_in_A)
        last = index_in_A
        
    right_seq.reverse()
    
    if left_seq == right_seq:
        print("No")
    else:
        print("Yes")

if __name__ == "__main__":
    main()