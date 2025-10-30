def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, L, R = map(int, data[:3])
    A = list(map(int, data[3:]))

    # For each A[i], the best X_i is A[i] clamped to [L, R].
    # That is: X_i = min(max(A[i], L), R).
    result = []
    for a in A:
        clamped_value = min(max(a, L), R)
        result.append(clamped_value)
    
    # Print the results separated by space
    print(*result)

# Do not forget to call main() here
main()