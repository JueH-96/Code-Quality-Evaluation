import sys
import threading

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Compute the earliest positions L[k] in A matching B[0..k]
    L = [0] * M
    ptr = 0
    for i, a in enumerate(A):
        if ptr < M and a == B[ptr]:
            L[ptr] = i
            ptr += 1
            if ptr == M:
                break
    if ptr < M:
        # B cannot be matched even once
        print("No")
        return

    # Compute the latest positions R[k] in A matching B[k..M-1]
    R = [0] * M
    ptr = M - 1
    for i in range(N - 1, -1, -1):
        if ptr >= 0 and A[i] == B[ptr]:
            R[ptr] = i
            ptr -= 1
            if ptr < 0:
                break
    if ptr >= 0:
        # Shouldn't really happen if forward scan succeeded, but safe guard
        print("No")
        return

    # If for any k we have L[k] < R[k], we can choose two different positions
    # for B[k], yielding two distinct subsequences.
    for k in range(M):
        if L[k] < R[k]:
            print("Yes")
            return

    # Otherwise the matching positions are unique
    print("No")

if __name__ == "__main__":
    main()