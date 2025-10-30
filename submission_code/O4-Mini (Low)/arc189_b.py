import sys
import threading

def main():
    import sys

    input = sys.stdin.readline
    N = int(input().strip())
    X = list(map(int, input().split()))
    # Compute the differences di = X[i] - X[i-1] for i=1..N-1 (1-based indices)
    # We'll index di in a 0-based list, where di_list[i-1] = X[i] - X[i-1]
    di = [X[i] - X[i-1] for i in range(1, N)]
    # We can apply operations that effectively allow us to reorder
    # di among indices of the same parity (i mod 2), sorting each parity-block
    # in ascending order to minimize the weighted sum.
    # Partition di by parity of their (1-based) index i
    di_odd = []   # holds di for i=1,3,5,... i.e. indices 0,2,4 in 0-based di
    di_even = []  # holds di for i=2,4,6,... i.e. indices 1,3,5 in 0-based di
    for idx, val in enumerate(di):
        # idx is 0-based for di, corresponds to i = idx+1
        if (idx + 1) & 1:
            di_odd.append(val)
        else:
            di_even.append(val)
    # Sort each sub‐list ascending
    di_odd.sort()
    di_even.sort()
    # Re‐assign back into a new di' array in increasing i order
    new_di = [0] * (N-1)
    o = 0
    e = 0
    for idx in range(N-1):
        if (idx + 1) & 1:
            new_di[idx] = di_odd[o]
            o += 1
        else:
            new_di[idx] = di_even[e]
            e += 1
    # Reconstruct the new X positions cumulatively and sum them
    total = X[0]
    curr = X[0]
    for d in new_di:
        curr += d
        total += curr
    print(total)

if __name__ == "__main__":
    main()