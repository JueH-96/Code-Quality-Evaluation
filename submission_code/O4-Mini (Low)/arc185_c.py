import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, X = map(int, input().split())
    A = list(map(int, input().split()))
    # Keep original indices
    B = [(val, idx+1) for idx, val in enumerate(A)]
    B.sort(key=lambda x: x[0])

    # We will fix one index k and do two‐sum on the prefix [0..k-1] with two pointers.
    for k in range(N-1, 1, -1):
        target = X - B[k][0]
        i = 0
        j = k - 1
        # two‐pointer search for B[i].val + B[j].val == target
        while i < j:
            s = B[i][0] + B[j][0]
            if s == target:
                # Found a triple
                idx1 = B[i][1]
                idx2 = B[j][1]
                idx3 = B[k][1]
                # must print in increasing order of indices
                res = sorted((idx1, idx2, idx3))
                print(res[0], res[1], res[2])
                return
            elif s < target:
                i += 1
            else:
                j -= 1

    # If we fall through, no solution
    print(-1)

if __name__ == "__main__":
    main()