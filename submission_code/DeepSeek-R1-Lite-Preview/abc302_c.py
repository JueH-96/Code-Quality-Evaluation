def differ_by_one(s1, s2):
    diff = 0
    for a, b in zip(s1, s2):
        if a != b:
            diff += 1
            if diff > 1:
                return False
    return diff == 1

def main():
    import sys
    sys.setrecursionlimit(10000)
    N, M = map(int, sys.stdin.readline().split())
    strings = [sys.stdin.readline().strip() for _ in range(N)]
    
    # Build adjacency matrix
    adjacency = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j and differ_by_one(strings[i], strings[j]):
                adjacency[i][j] = True
    
    # Backtracking to find Hamiltonian path
    def backtrack(current, visited):
        if visited == (1 << N) - 1:
            return True
        for neighbor in range(N):
            if adjacency[current][neighbor] and not (visited & (1 << neighbor)):
                if backtrack(neighbor, visited | (1 << neighbor)):
                    return True
        return False
    
    # Try starting from each node
    for start in range(N):
        if backtrack(start, 1 << start):
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()