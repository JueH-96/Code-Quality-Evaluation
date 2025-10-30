import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    N = int(line[0])
    Tprime = line[1].rstrip('
')
    n = len(Tprime)
    res = []
    # Helper to check if A can become B by at most one insert/delete/substitute
    for i in range(1, N+1):
        A = data.readline().rstrip('
')
        m = len(A)
        # If length difference > 1, impossible
        if abs(m - n) > 1:
            continue
        # Case 1: same length -> equal or one substitution
        if m == n:
            diff = 0
            for ca, cb in zip(A, Tprime):
                if ca != cb:
                    diff += 1
                    if diff > 1:
                        break
            if diff <= 1:
                res.append(i)
        # Case 2: A shorter by 1 -> one insertion in A makes Tprime
        elif m + 1 == n:
            # allow one skip in Tprime
            skip_used = False
            ia = 0
            ib = 0
            while ia < m and ib < n:
                if A[ia] == Tprime[ib]:
                    ia += 1
                    ib += 1
                else:
                    if skip_used:
                        break
                    skip_used = True
                    ib += 1
            else:
                # loop ended normally; it's ok even if pointers not at end
                res.append(i)
        # Case 3: A longer by 1 -> one deletion from A makes Tprime
        else:  # m == n + 1
            skip_used = False
            ia = 0
            ib = 0
            while ia < m and ib < n:
                if A[ia] == Tprime[ib]:
                    ia += 1
                    ib += 1
                else:
                    if skip_used:
                        break
                    skip_used = True
                    ia += 1
            else:
                res.append(i)

    # Output
    out = sys.stdout
    out.write(str(len(res)) + "
")
    if res:
        out.write(" ".join(map(str, res)) + "
")

if __name__ == "__main__":
    main()