def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    i = 0
    while i < len(A) - 1:
        if abs(A[i] - A[i+1]) == 1:
            i += 1
            continue
        
        if A[i] < A[i+1]:
            # Insert increasing sequence
            start = A[i] + 1
            end = A[i+1]
            A = A[:i+1] + list(range(start, end)) + A[i+1:]
        else:
            # Insert decreasing sequence
            start = A[i] - 1
            end = A[i+1]
            A = A[:i+1] + list(range(start, end, -1)) + A[i+1:]
        
        # No need to increment i, because we need to check the newly inserted part again
    
    print(" ".join(map(str, A)))

if __name__ == "__main__":
    main()