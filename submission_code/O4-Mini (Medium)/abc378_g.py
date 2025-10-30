import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10000)
    A,B,M = map(int, sys.stdin.readline().split())
    N = A*B - 1
    # compute factorials mod M
    fact = [1]*(N+1)
    for i in range(1,N+1):
        fact[i] = fact[i-1]*i % M
    # build the shape λ = (B, B, ..., B, B-1) length A
    # compute hook‐lengths and their product
    hooks = []
    # λ_i = B for i< A, λ_A = B-1
    def lam(i):
        return B if i< A else B-1
    for i in range(1,A+1):
        li = lam(i)
        for j in range(1,li+1):
            # hook = # to right + # below + 1
            right = lam(i) - j
            below = 0
            for k in range(i+1, A+1):
                if lam(k) >= j:
                    below += 1
            hooks.append((right + below + 1) % M)
    prod_hooks = 1
    for h in hooks:
        prod_hooks = prod_hooks * h % M
    # f = number of SYT of shape λ
    inv = pow(prod_hooks, M-2, M)
    f = fact[N] * inv % M
    # total permutations of shape λ under RSK = f^2
    # Now impose the extra condition:
    #   P(1,B) < P(A,1)
    # In the special shape λ, the bottom‐right missing corner is at (A,B).
    # There are exactly two removable corners of λ:
    #   c1 = (A, B-1)   and   c2 = (A-1, B).
    # One checks by the RSK‐theory argument that
    # the required gap‐condition forces the largest label N
    # to occupy the corner c1 = (A, B-1).
    # Hence the count of valid insertion‐tableaux P is
    #   f1 = #SYT of shape λ with N at c1
    #       = f(λ minus c1)
    # and each such P admits f choices of recording‐tableau Q.
    # We compute f1 by removing the cell (A, B-1) and
    # applying the hook‐formula again on the smaller shape.
    # Build hook‐product for λ' = λ \ {(A, B-1)} of size N-1.
    hooks2 = []
    def lam2(i):
        # in row A, length is B-1; removing one more gives B-2
        if i< A:
            return B
        else:
            return B-2
    for i in range(1,A+1):
        li = lam2(i)
        for j in range(1,li+1):
            right = lam2(i) - j
            below = 0
            for k in range(i+1, A+1):
                if lam2(k) >= j:
                    below += 1
            hooks2.append((right + below + 1) % M)
    prod_hooks2 = 1
    for h in hooks2:
        prod_hooks2 = prod_hooks2 * h % M
    f1 = fact[N-1] * pow(prod_hooks2, M-2, M) % M
    # total valid permutations = f1 * f  mod M
    ans = f1 * f % M
    print(ans)

if __name__ == "__main__":
    main()