# YOUR CODE HERE
import sys

def main():
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    n = len(S)
    S_current = list(S)

    positions_to_fix = [i for i in range(n) if S[i] != T[i]]

    X = []

    if not positions_to_fix:
        print(0)
        return

    while positions_to_fix:
        candidates = []
        for p in positions_to_fix:
            S_new = S_current.copy()
            S_new[p] = T[p]
            candidates.append(( ''.join(S_new), p))
        # Select the lex smallest S_new
        candidates.sort()
        S_new_str, p_fixed = candidates[0]
        S_current = list(S_new_str)
        X.append(S_new_str)
        positions_to_fix.remove(p_fixed)

    print(len(X))
    for s in X:
        print(s)

if __name__ == "__main__":
    main()