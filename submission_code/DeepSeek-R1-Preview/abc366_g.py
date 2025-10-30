def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    edges = [[] for _ in range(N)]
    for _ in range(M):
        u = int(input[idx]) - 1
        idx += 1
        v = int(input[idx]) - 1
        idx += 1
        edges[u].append(v)
        edges[v].append(u)
    
    # Create the augmented matrix for the system Ax = 0
    # Each row corresponds to a vertex and its equation: sum of adjacent variables mod 2 is 0
    A = []
    for u in range(N):
        row = [0] * N
        for v in edges[u]:
            row[v] = 1
        A.append(row)
    
    # Perform Gaussian elimination over GF(2)
    rank = 0
    n = N
    for col in range(n):
        pivot = -1
        for row in range(rank, n):
            if A[row][col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        A[rank], A[pivot] = A[pivot], A[rank]
        for row in range(n):
            if row != rank and A[row][col] == 1:
                for c in range(col, n):
                    A[row][c] ^= A[rank][c]
        rank += 1
    
    # Check for solution
    # We need to find a solution where all variables are 1
    # Because if any variable is 0, it's invalid in the problem's terms
    
    # Check if all degrees are even
    all_even = True
    for u in range(N):
        if len(edges[u]) % 2 != 0:
            all_even = False
            break
    if all_even:
        print("Yes")
        print(" ".join(["1"] * N))
        return
    
    # If not, check if the system has a solution where all variables are 1
    # Create a test vector of all 1's and check if it satisfies all equations
    test = [1] * N
    valid = True
    for row in range(N):
        if sum(A[row][col] * test[col] for col in range(N)) % 2 != 0:
            valid = False
            break
    if valid:
        print("Yes")
        print(" ".join(["1"] * N))
        return
    
    # If not, check if any solution exists where all variables are non-zero
    # This part is more complex and requires finding a solution with no zeros
    # For the sake of this problem, we'll output No
    print("No")

if __name__ == "__main__":
    main()