import sys

def main():
    input = sys.stdin.readline
    data = input().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, input().split()))
    # Compute L[i]: max length of increasing-by-1-from-1 segment ending at i
    C = [A[i] + (i+1) for i in range(N)]
    idxs = list(range(N))
    idxs.sort(key=C.__getitem__)
    L = [0] * N
    maxj = -1
    p = 0
    for i in range(N):
        thresh = i + 1
        # include all j with C[j] <= thresh
        while p < N and C[idxs[p]] <= thresh:
            j = idxs[p]
            if j > maxj:
                maxj = j
            p += 1
        L[i] = i - maxj

    # Compute R[i] similarly on reversed array
    AR = A[::-1]
    Cr = [AR[i] + (i+1) for i in range(N)]
    idxs = list(range(N))
    idxs.sort(key=Cr.__getitem__)
    Lr = [0] * N
    maxj = -1
    p = 0
    for i in range(N):
        thresh = i + 1
        while p < N and Cr[idxs[p]] <= thresh:
            j = idxs[p]
            if j > maxj:
                maxj = j
            p += 1
        Lr[i] = i - maxj

    # Combine L and reversed Lr to get the pyramid size at each center
    ans = 0
    for i in range(N):
        # R[i] is Lr at mirrored position
        ri = Lr[N-1-i]
        ki = L[i] if L[i] < ri else ri
        if ki > ans:
            ans = ki

    print(ans)

if __name__ == "__main__":
    main()