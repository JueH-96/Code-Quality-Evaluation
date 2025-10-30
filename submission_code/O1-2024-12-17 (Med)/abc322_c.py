def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Pointer to the current index of the fireworks days
    j = 0
    
    for i in range(1, N+1):
        # Advance j if the current fireworks day is before day i
        while j < M and A[j] < i:
            j += 1
        print(A[j] - i)

# Do not forget to call main() at the end.
main()