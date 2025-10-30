def solve():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    
    # We will store the combination results in a 2D array A
    # where A[i][j] = A_{i,j} if i >= j, otherwise we access A[j][i].
    A = [[0]*(N+1) for _ in range(N+1)]
    
    idx = 1
    for i in range(1, N+1):
        # Read i values for the i-th line of input
        for j in range(1, i+1):
            A[i][j] = int(data[idx])
            idx += 1
    
    current_element = 1
    # Combine current_element with 1, then 2, up to N
    for x in range(1, N+1):
        if current_element >= x:
            current_element = A[current_element][x]
        else:
            current_element = A[x][current_element]
    
    print(current_element)

# Call solve() to execute
def main():
    solve()

if __name__ == "__main__":
    main()