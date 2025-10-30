import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # L[i]: the maximum possible height of a strictly increasing
    # ramp ending at i with step 1, but capped by A[i].
    L = [0] * N
    # R[i]: similar from the right side.
    R = [0] * N

    # Compute L
    # At index 0, the best we can do is height=1 (if A[0]>=1).
    # Then L[i] = min(A[i], L[i-1] + 1).
    # This means "we could build a ramp 1,2,... up to here" as high as A allows.
    L[0] = 1
    for i in range(1, N):
        # try to extend the ramp by 1, but no more than A[i].
        # also at least height 1.
        L[i] = min(A[i], L[i-1] + 1)

    # Compute R similarly, from the right end.
    R[N-1] = 1
    for i in range(N-2, -1, -1):
        R[i] = min(A[i], R[i+1] + 1)

    # At each center i, the largest pyramid "peak" we can form
    # is min(L[i], R[i]). We pick the maximum over i.
    ans = 0
    for i in range(N):
        peak = L[i] if L[i] < R[i] else R[i]
        if peak > ans:
            ans = peak

    # ans is the pyramid size k.
    print(ans)

if __name__ == "__main__":
    main()