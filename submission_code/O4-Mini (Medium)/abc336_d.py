import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    # Lcap[i]: the maximum height of an increasing-by-1 sequence ending at i,
    # when lowering A values as needed, i.e. min(Lcap[i-1]+1, A[i])
    Lcap = [0] * n
    prev = 0
    for i in range(n):
        # we can extend the previous run by +1, but cannot exceed A[i]
        cur = prev + 1
        if cur > A[i]:
            cur = A[i]
        Lcap[i] = cur
        prev = cur

    # Rcap[i]: similarly from the right
    Rcap = [0] * n
    prev = 0
    for i in range(n-1, -1, -1):
        cur = prev + 1
        if cur > A[i]:
            cur = A[i]
        Rcap[i] = cur
        prev = cur

    # The largest pyramid centered at i has size k_i = min(Lcap[i], Rcap[i]).
    # We take the maximum over all centers i.
    ans = 0
    for i in range(n):
        ki = Lcap[i]
        if Rcap[i] < ki:
            ki = Rcap[i]
        if ki > ans:
            ans = ki

    # The problem asks for k, the size of the pyramid.
    # Our ki is exactly that k.
    print(ans)

if __name__ == "__main__":
    main()