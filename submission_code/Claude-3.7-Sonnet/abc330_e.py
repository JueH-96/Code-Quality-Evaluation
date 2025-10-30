def main():
    N, Q = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Create a dictionary to keep track of values in range [0, N]
    # We only care about values in this range for MEX calculation
    count = {}
    for a in A:
        if 0 <= a <= N:
            count[a] = count.get(a, 0) + 1
    
    for _ in range(Q):
        i, x = map(int, input().split())
        i -= 1  # Convert to 0-indexed
        
        # Remove the old value from count
        old_value = A[i]
        if 0 <= old_value <= N:
            count[old_value] -= 1
            if count[old_value] == 0:
                del count[old_value]
        
        # Add the new value to count
        A[i] = x
        if 0 <= x <= N:
            count[x] = count.get(x, 0) + 1
        
        # Find the MEX (minimum excluded value)
        mex = 0
        while mex in count:
            mex += 1
        
        print(mex)

if __name__ == "__main__":
    main()