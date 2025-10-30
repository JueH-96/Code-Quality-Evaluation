import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(K)]
    # A is sorted increasing by input guarantee
    # If even number of lost socks, pair them greedily
    if K % 2 == 0:
        ans = 0
        # pair A[0]&A[1], A[2]&A[3], ...
        for i in range(0, K, 2):
            ans += A[i+1] - A[i]
        print(ans)
        return
    # K is odd: we must drop one lost sock and pair the rest greedily
    L = K
    M = L - 1  # length of difference array d_full
    # Build d_full[i] = A[i+1] - A[i] for i in [0..M-1]
    d_full = [0] * M
    for i in range(M):
        d_full[i] = A[i+1] - A[i]
    # Build prefix sums of even and odd indexed differences
    # pref_even[i] = sum of d_full[j] for j in [0..i-1] and j%2==0
    # pref_odd[i]  = sum of d_full[j] for j in [0..i-1] and j%2==1
    pref_even = [0] * (M + 1)
    pref_odd  = [0] * (M + 1)
    for i in range(1, M+1):
        # we consider d_full[i-1]
        pref_even[i] = pref_even[i-1]
        pref_odd[i]  = pref_odd[i-1]
        if (i-1) & 1 == 0:
            pref_even[i] += d_full[i-1]
        else:
            pref_odd[i] += d_full[i-1]
    total_odd = pref_odd[M]
    INF = 10**30
    ans = INF
    # removal at j even (0-based): cost = sum even-diffs before j + sum odd-diffs after j
    # cost_j = pref_even[j] + (total_odd - pref_odd[j+1])
    # if j+1 > M then pref_odd[j+1] = total_odd
    for j in range(0, L, 2):
        # sum of evens i<j
        left = pref_even[j]
        # sum of odds i>=j+1..M-1
        if j+1 <= M:
            right = total_odd - pref_odd[j+1]
        else:
            right = 0
        cost = left + right
        if cost < ans:
            ans = cost
    # removal at j odd: cost = sum even-diffs before j-1 + (A[j+1]-A[j-1]) + sum odd-diffs after j+1
    # cost_j = pref_even[j-1] + (A[j+1]-A[j-1]) + (total_odd - pref_odd[j+2])
    # if j+2 > M then pref_odd[j+2] = total_odd
    for j in range(1, L, 2):
        # j+1 < L always since L is odd and last index L-1 is even
        left = pref_even[j-1]
        mid = A[j+1] - A[j-1]
        if j+2 <= M:
            right = total_odd - pref_odd[j+2]
        else:
            right = 0
        cost = left + mid + right
        if cost < ans:
            ans = cost
    print(ans)

if __name__ == "__main__":
    main()