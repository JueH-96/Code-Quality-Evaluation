import sys
import threading

def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    # Find the earliest (leftmost) match positions L for B in A
    L = [0]*M
    k = 0
    for i in range(N):
        if k < M and A[i] == B[k]:
            L[k] = i
            k += 1
            if k == M:
                break
    # If we didn't match the whole B, no subsequence at all
    if k < M:
        print("No")
        return
    
    # Find the latest (rightmost) match positions R for B in A
    R = [0]*M
    k = M-1
    for i in range(N-1, -1, -1):
        if k >= 0 and A[i] == B[k]:
            R[k] = i
            k -= 1
            if k < 0:
                break
    
    # If L and R coincide exactly, there's exactly one way.
    # Otherwise there are at least two distinct subsequences.
    if L == R:
        print("No")
    else:
        print("Yes")

if __name__ == "__main__":
    main()