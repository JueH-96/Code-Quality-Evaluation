import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    K = int(data[0])
    S = data[1]
    T = data[2]
    n = len(S)
    m = len(T)
    # If the difference in lengths is greater than K, impossible
    if abs(n - m) > K:
        print("No")
        return
    # Quick check
    if S == T:
        print("Yes")
        return
    # Myers O((N+M)K) algorithm, only check <= K edits
    # V[k] = furthest x reached on diagonal k = x - y
    # We'll index V with offset = K, so array size = 2*K+1
    offset = K
    size = 2*K + 1
    V = [-1] * size
    # d = 0 init
    V[offset + 0] = 0
    # target diagonal
    target_k = n - m
    # For d from 0..K
    for d in range(0, K+1):
        # for all k in [-d, -d+2, ..., d]
        for k in range(-d, d+1, 2):
            idx = k + offset
            # choose: insertion (down) or deletion (right)
            if k == -d:
                # must come down: take from k+1
                x_start = V[idx + 1]
            elif k == d:
                # must come right: take from k-1
                x_start = V[idx - 1] + 1
            else:
                # pick larger x
                if V[idx - 1] + 1 > V[idx + 1]:
                    x_start = V[idx - 1] + 1
                else:
                    x_start = V[idx + 1]
            y_start = x_start - k
            # follow diagonal (matches)
            x = x_start
            y = y_start
            # advance while match
            while x < n and y < m and S[x] == T[y]:
                x += 1
                y += 1
            V[idx] = x
            # check if reached end
            if k == target_k and x >= n and y >= m:
                print("Yes")
                return
        # end for k
    # end for d
    print("No")

if __name__ == "__main__":
    main()