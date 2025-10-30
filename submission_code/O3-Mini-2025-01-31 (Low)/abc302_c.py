def main():
    import sys
    input = sys.stdin.readline
    N, M = map(int, input().split())
    strings = [input().strip() for _ in range(N)]
    
    # Helper function: check if two strings differ by exactly one char.
    def is_valid_adj(s1, s2):
        diff = 0
        for a, b in zip(s1, s2):
            if a != b:
                diff += 1
            if diff > 1:
                return False
        return diff == 1
    
    # We'll use backtracking to generate permutations and check adjacent string difference
    used = [False] * N
    perm = [None] * N
    found = [False]  # using list for closure
    
    def dfs(depth):
        if found[0]:
            return
        if depth == N:
            found[0] = True
            return
        for i in range(N):
            if not used[i]:
                if depth > 0 and not is_valid_adj(perm[depth-1], strings[i]):
                    continue
                used[i] = True
                perm[depth] = strings[i]
                dfs(depth+1)
                if found[0]:
                    return
                used[i] = False
                
    dfs(0)
    
    print("Yes" if found[0] else "No")

if __name__ == '__main__':
    main()