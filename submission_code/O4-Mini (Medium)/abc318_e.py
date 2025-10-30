import sys
import threading

def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    # positions[v] will hold the list of indices where A[i] == v
    positions = [[] for _ in range(n+1)]
    for idx, v in enumerate(A):
        positions[v].append(idx)
    ans = 0
    # For each value v, compute its contribution
    # Let P = positions[v], m = len(P)
    # We need sum_{0 <= i < j < m} (P[j] - P[i] - (j - i))
    # This equals sum_i (P[i] - i) * (2*i - m + 1)
    for P in positions:
        m = len(P)
        if m < 2:
            continue
        # Accumulate for this value
        # delta_i = P[i] - i
        # coef   = 2*i - m + 1
        # contribution = sum delta_i * coef
        for i, Pi in enumerate(P):
            delta = Pi - i
            coef = 2 * i - m + 1
            ans += delta * coef
    print(ans)

if __name__ == "__main__":
    main()