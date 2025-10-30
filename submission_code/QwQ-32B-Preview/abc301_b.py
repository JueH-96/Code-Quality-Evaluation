def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    while True:
        insertion_made = False
        for i in range(len(A) - 1):
            if abs(A[i] - A[i+1]) != 1:
                # Need to insert between A[i] and A[i+1]
                if A[i] < A[i+1]:
                    insert_list = list(range(A[i] + 1, A[i+1]))
                else:
                    insert_list = list(range(A[i] - 1, A[i+1], -1))
                A[i+1:i+1] = insert_list
                insertion_made = True
                break  # Start over to check from the beginning
        if not insertion_made:
            break  # No insertions were made, terminate
        
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()