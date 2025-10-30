def main():
    import sys
    data = sys.stdin.read().split()
    if not data: 
        return
    N = int(data[0])
    M = int(data[1])
    threshold = 10**9
    result = 0
    term = 1  # Represents N^i for the current i
    
    for i in range(M + 1):
        result += term
        if result > threshold:
            print("inf")
            return
        # If this is not the last iteration, update term for next power
        if i < M:
            term *= N
            # Optimization: For N > 1, if the next term itself is already above threshold,
            # adding it to result will definitely exceed threshold.
            if N > 1 and term > threshold:
                print("inf")
                return
    print(result)

if __name__ == '__main__':
    main()