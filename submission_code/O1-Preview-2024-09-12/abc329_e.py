# YOUR CODE HERE
import sys
import threading

sys.setrecursionlimit(1 << 25)

def main():
    N, M = map(int, input().split())
    S = input()
    T = input()
    N = len(S)
    M = len(T)
    
    intervals = []
    for l in range(N - M + 1):
        if S[l:l+M] == T:
            intervals.append((l, l+M-1))
    
    if not intervals:
        print('No')
        return
    
    # Merge intervals and check if they cover [0, N-1]
    intervals.sort()
    merged = []
    start, end = intervals[0]
    for s, e in intervals[1:]:
        if s <= end + 1:
            end = max(end, e)
        else:
            merged.append((start, end))
            start, end = s, e
    merged.append((start, end))
    
    covered = 0
    for s, e in merged:
        if s > covered:
            print('No')
            return
        covered = max(covered, e +1)
    
    if covered >= N:
        print('Yes')
    else:
        print('No')
    
threading.Thread(target=main).start()