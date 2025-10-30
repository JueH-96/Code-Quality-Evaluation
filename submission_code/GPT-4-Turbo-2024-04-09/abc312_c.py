def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    # Sort the seller and buyer lists
    A.sort()
    B.sort()
    
    # We need to find the minimum X such that:
    # Number of sellers who can sell at X yen >= Number of buyers who can buy at X yen
    
    # We can use a two-pointer technique to find the smallest X that satisfies the condition
    # We will iterate over possible X values starting from the smallest possible value that could be considered
    
    i, j = 0, 0
    min_X = 0
    
    # We need to find the smallest X such that sellers >= buyers
    while i < N and j < M:
        if A[i] <= B[j]:
            # A[i] is a candidate for X
            # Check if the number of sellers from i to N is greater than or equal to the number of buyers from j to M
            if (N - i) >= (M - j):
                min_X = A[i]
                break
            i += 1
        else:
            j += 1
    
    # If we exit the loop without setting min_X, it means we need to consider the next possible X after the last A[i]
    if min_X == 0:
        if i < N:
            min_X = A[i]
        else:
            # If all sellers are considered and we still don't have enough, we need to go beyond the maximum B[j]
            min_X = B[-1] + 1
    
    print(min_X)

if __name__ == "__main__":
    main()