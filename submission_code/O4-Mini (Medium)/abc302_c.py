import sys
import itertools

def main():
    input_data = sys.stdin.read().split()
    N, M = map(int, input_data[:2])
    S = input_data[2:]
    # Build adjacency: adj[i][j] = True iff S[i] and S[j] differ by exactly one char
    adj = [[False]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                diff = 0
                for a, b in zip(S[i], S[j]):
                    if a != b:
                        diff += 1
                        if diff > 1:
                            break
                if diff == 1:
                    adj[i][j] = True

    # Check all permutations for a Hamiltonian path
    for perm in itertools.permutations(range(N)):
        ok = True
        for k in range(N-1):
            if not adj[perm[k]][perm[k+1]]:
                ok = False
                break
        if ok:
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    main()