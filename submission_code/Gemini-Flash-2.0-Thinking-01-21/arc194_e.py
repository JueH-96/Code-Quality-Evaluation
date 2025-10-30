import sys

def solve():
    N, X, Y = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Count the number of '1's in S and T at indices k such that k % X == r
    # for each residue r from 0 to X-1.
    # The operations 0^X 1^Y <-> 1^Y 0^X effectively swap the positions of '1's.
    # Operation A moves '1's from indices [i+X, i+X+Y-1] to [i, i+Y-1].
    # An index k in [i+X, i+X+Y-1] corresponds to k-X in [i, i+Y-1].
    # k % X = (k-X) % X. The residue modulo X is preserved.
    # Operation B moves '1's from indices [i, i+Y-1] to [i+X, i+X+Y-1].
    # An index k in [i, i+Y-1] corresponds to k+X in [i+X, i+X+Y-1].
    # k % X = (k+X) % X. The residue modulo X is preserved.
    # Therefore, the number of '1's at indices congruent to r modulo X must be
    # the same in S and T for the transformation to be possible.

    countS_mod_X = [0] * X
    countT_mod_X = [0] * X

    for k in range(N):
        if S[k] == '1':
            countS_mod_X[k % X] += 1
        if T[k] == '1':
            countT_mod_X[k % X] += 1

    # Check if the counts modulo X are the same for all residues
    if countS_mod_X == countT_mod_X:
        print("Yes")
    else:
        print("No")

solve()