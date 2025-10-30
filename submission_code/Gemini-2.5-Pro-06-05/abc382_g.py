import sys

def get_tile(x, y, K):
    """
    Finds the tile (i, j, k) containing the point (x + 0.5, y + 0.5).
    We can use x and y directly since they are integers and K >= 2.
    """
    i = x // K
    j = y // K
    # Python's // is floor division and % for negative numbers works as expected for positive K.
    if (i + j) % 2 == 0:  # Same parity -> Horizontal tiles
        k = y % K
    else:  # Different parity -> Vertical tiles
        k = x % K
    return i, j, k

def solve():
    """
    Solves a single test case.
    """
    try:
        K, Sx, Sy, Tx, Ty = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        return

    is_, js, ks = get_tile(Sx, Sy, K)
    it, jt, kt = get_tile(Tx, Ty, K)

    if is_ == it and js == jt:
        print(abs(ks - kt))
        return

    ans = float('inf')

    # Path 1: (is, js) -> (it, js) -> (it, jt) (Horizontal then Vertical)
    k, cost, k_free = ks, 0, False
    
    # Horizontal segment
    if is_ != it:
        di = 1 if it > is_ else -1
        for i in range(is_, it, di):
            par = (i + js) % 2
            if di == 1:  # move right
                if par == 0:  # From Horizontal block (easy)
                    cost += 1
                    k = 0
                    k_free = False
                else:  # From Vertical block (hard)
                    cost += 1 if k_free else abs(k - (K - 1)) + 1
                    k_free = True
            else:  # move left
                if par == 0:  # From Horizontal block (easy)
                    cost += 1
                    k = K - 1
                    k_free = False
                else:  # From Vertical block (hard)
                    cost += 1 if k_free else abs(k - 0) + 1
                    k_free = True
    
    # Vertical segment
    if js != jt:
        dj = 1 if jt > js else -1
        for j in range(js, jt, dj):
            par = (it + j) % 2
            if dj == 1:  # move up
                if par == 0:  # From Horizontal block (hard)
                    cost += 1 if k_free else abs(k - (K - 1)) + 1
                    k_free = True
                else:  # From Vertical block (easy)
                    cost += 1
                    k = 0
                    k_free = False
            else:  # move down
                if par == 0:  # From Horizontal block (hard)
                    cost += 1 if k_free else abs(k - 0) + 1
                    k_free = True
                else:  # From Vertical block (easy)
                    cost += 1
                    k = K - 1
                    k_free = False

    cost += 0 if k_free else abs(k - kt)
    ans = min(ans, cost)

    # Path 2: (is, js) -> (is, jt) -> (it, jt) (Vertical then Horizontal)
    k, cost, k_free = ks, 0, False
    
    # Vertical segment
    if js != jt:
        dj = 1 if jt > js else -1
        for j in range(js, jt, dj):
            par = (is_ + j) % 2
            if dj == 1:  # move up
                if par == 0:  # From H (hard)
                    cost += 1 if k_free else abs(k - (K - 1)) + 1
                    k_free = True
                else:  # From V (easy)
                    cost += 1
                    k = 0
                    k_free = False
            else:  # move down
                if par == 0:  # From H (hard)
                    cost += 1 if k_free else abs(k - 0) + 1
                    k_free = True
                else:  # From V (easy)
                    cost += 1
                    k = K - 1
                    k_free = False
                    
    # Horizontal segment
    if is_ != it:
        di = 1 if it > is_ else -1
        for i in range(is_, it, di):
            par = (i + jt) % 2
            if di == 1:  # move right
                if par == 0:  # From H (easy)
                    cost += 1
                    k = 0
                    k_free = False
                else:  # From V (hard)
                    cost += 1 if k_free else abs(k - (K - 1)) + 1
                    k_free = True
            else:  # move left
                if par == 0:  # From H (easy)
                    cost += 1
                    k = K - 1
                    k_free = False
                else:  # From V (hard)
                    cost += 1 if k_free else abs(k - 0) + 1
                    k_free = True

    cost += 0 if k_free else abs(k - kt)
    ans = min(ans, cost)
    
    print(ans)


def main():
    """
    Main function to read the number of test cases and run the solver for each.
    """
    try:
        T_str = sys.stdin.readline()
        if not T_str: return
        T = int(T_str)
        for _ in range(T):
            solve()
    except (IOError, ValueError):
        # Handles potential empty input or parsing errors gracefully
        return

if __name__ == "__main__":
    main()