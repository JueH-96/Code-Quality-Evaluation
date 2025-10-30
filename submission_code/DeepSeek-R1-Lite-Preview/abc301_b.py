def main():
    import sys
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    
    while True:
        insertion_done = False
        for i in range(len(A) - 1):
            if abs(A[i] - A[i+1]) != 1:
                if A[i] < A[i+1]:
                    insert_numbers = list(range(A[i]+1, A[i+1]))
                else:
                    insert_numbers = list(range(A[i]-1, A[i+1], -1))
                A = A[:i+1] + insert_numbers + A[i+1:]
                insertion_done = True
                break
        if not insertion_done:
            break
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()