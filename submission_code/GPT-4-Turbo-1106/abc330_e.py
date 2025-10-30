from collections import Counter

def main():
    # Read the number of elements and queries
    N, Q = map(int, input().split())
    
    # Read the sequence A
    A = list(map(int, input().split()))
    
    # Create a counter for the elements in A
    counter = Counter(A)
    
    # Initialize the mex value
    mex = 0
    while counter[mex]:
        mex += 1
    
    # Process each query
    for _ in range(Q):
        i, x = map(int, input().split())
        i -= 1  # Convert to 0-based index
        
        # Decrease the count of the old value
        counter[A[i]] -= 1
        # If the old value was equal to current mex and it's no longer in A, update mex
        if A[i] == mex and counter[A[i]] == 0:
            while counter[mex]:
                mex += 1
        
        # Set the new value and increase its count
        A[i] = x
        counter[x] += 1
        # If the new value is less than current mex, update mex
        if x < mex:
            mex = x
            while counter[mex]:
                mex += 1
        
        # Print the current mex
        print(mex)

if __name__ == "__main__":
    main()