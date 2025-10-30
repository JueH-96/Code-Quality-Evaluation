def main():
    import sys
    from collections import deque

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    X = int(data[1]) - 1  # Adjust X to 0-based index
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+2*N]))
    P = list(map(lambda x: int(x)-1, data[2+2*N:2+3*N]))
    Q = list(map(lambda x: int(x)-1, data[2+3*N:2+4*N]))

    # Function to compute levels for permutation P or Q
    def compute_levels(permutation, X):
        levels = [-1] * N
        levels[X] = 0
        queue = deque([X])
        while queue:
            current = queue.popleft()
            parent = permutation[current]
            if levels[parent] == -1:
                levels[parent] = levels[current] + 1
                queue.append(parent)
        return levels

    # Compute levels for red balls via P
    levels_red = compute_levels(P, X)
    # Compute levels for blue balls via Q
    levels_blue = compute_levels(Q, X)

    # Find the maximum level needed for boxes with balls
    max_operations = -1
    for i in range(N):
        if A[i]:
            if levels_red[i] == -1:
                print(-1)
                return
            max_operations = max(max_operations, levels_red[i])
        if B[i]:
            if levels_blue[i] == -1:
                print(-1)
                return
            max_operations = max(max_operations, levels_blue[i])

    print(max_operations)

if __name__ == "__main__":
    main()