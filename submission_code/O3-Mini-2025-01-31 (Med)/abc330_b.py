def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # First three entries are N, L, and R.
    N = int(input_data[0])
    L = int(input_data[1])
    R = int(input_data[2])
    
    # The rest are the sequence A.
    A = list(map(int, input_data[3:]))
    
    # For each element A[i], the optimal X is A[i] clamped to the range [L, R].
    # If A[i] is less than L then choose L, if A[i] is greater than R then choose R,
    # otherwise just choose A[i].
    result = []
    for a in A:
        if a < L:
            result.append(L)
        elif a > R:
            result.append(R)
        else:
            result.append(a)
    
    # Print the results separated by spaces.
    print(" ".join(map(str, result)))

if __name__ == '__main__':
    main()