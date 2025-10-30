def main():
    import sys
    input = sys.stdin.readline
    from itertools import permutations

    N, M = map(int, input().split())
    S = [input().strip() for _ in range(N)]

    # Precompute whether two strings have Hamming distance exactly 1
    def hamming_dist_one(a, b):
        diff = 0
        for x, y in zip(a, b):
            if x != y:
                diff += 1
                if diff > 1:
                    return False
        return diff == 1

    adj = [[False]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j and hamming_dist_one(S[i], S[j]):
                adj[i][j] = True

    # Try all permutations of indices
    for perm in permutations(range(N)):
        ok = True
        for i in range(N-1):
            if not adj[perm[i]][perm[i+1]]:
                ok = False
                break
        if ok:
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    main()