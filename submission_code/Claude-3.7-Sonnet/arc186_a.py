def is_valid(N, K):
    """
    Check if a matrix with exactly K fixed elements can exist for an N×N matrix.
    Fixed elements satisfy the formula r * (2 * N - r) for some integer r.
    """
    for r in range(N + 1):
        if r * (2 * N - r) == K:
            return True
    return False

def main():
    # Read input
    N, Q = map(int, input().split())
    results = []
    
    # Process each query
    for _ in range(Q):
        K = int(input())
        if is_valid(N, K):
            results.append("Yes")
        else:
            results.append("No")
    
    # Print results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()