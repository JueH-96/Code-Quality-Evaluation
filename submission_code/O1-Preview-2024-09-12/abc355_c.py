# YOUR CODE HERE
import sys

import threading

def main():
    import sys
    input = sys.stdin.readline

    N, T = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    rows = [0]*N
    cols = [0]*N
    diag1 = 0
    diag2 = 0
    marked = set()

    for t in range(T):
        a = A[t]
        a -=1  # Convert to zero-based index of cells from 0 to N^2-1
        i = a // N
        j = a % N

        if (i,j) in marked:
            continue
        marked.add((i,j))

        rows[i] +=1
        if rows[i]==N:
            print(t+1)
            return
        cols[j] +=1
        if cols[j]==N:
            print(t+1)
            return

        if i == j:
            diag1 +=1
            if diag1==N:
                print(t+1)
                return
        if i + j == N -1:
            diag2 +=1
            if diag2==N:
                print(t+1)
                return
    print(-1)
threading.Thread(target=main).start()