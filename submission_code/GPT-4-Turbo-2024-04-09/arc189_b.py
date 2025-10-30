def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = list(map(int, data[1:]))
    
    # Since the operation only affects the middle two elements of any four consecutive elements,
    # and the operation is symmetric around the midpoint of the outer two elements,
    # the minimum sum of coordinates is achieved by:
    # - Keeping the first and last elements fixed
    # - For every group of four, the inner two elements are symmetric around the midpoint of the outer two
    
    # The sum of the coordinates of the pieces after optimal operations:
    # - The sum of the first and last elements remains the same
    # - For each group of four from i to i+3:
    #   - The sum of the positions of the i+1 and i+2 elements becomes twice the midpoint of i and i+3
    
    # The sum of the coordinates of the pieces:
    sum_X = sum(X)
    
    # We need to find the minimum possible sum after operations
    # The optimal sum can be calculated directly:
    # - The first and last elements always remain the same
    # - For each possible i (from 0 to N-4), the i+1 and i+2 elements will be positioned symmetrically
    #   around the midpoint of X[i] and X[i+3]
    
    # The sum of the coordinates after optimal operations:
    # - X[0] + X[N-1] (first and last elements)
    # - For each group of four from i to i+3:
    #   - The sum of the positions of the i+1 and i+2 elements becomes twice the midpoint of i and i+3
    #   - This is 2 * (X[i] + X[i+3]) / 2 = X[i] + X[i+3]
    
    # The sum of the coordinates after optimal operations:
    optimal_sum = X[0] + X[N-1]
    for i in range(N-3):
        optimal_sum += X[i] + X[i+3]
    
    print(optimal_sum)

if __name__ == "__main__":
    main()