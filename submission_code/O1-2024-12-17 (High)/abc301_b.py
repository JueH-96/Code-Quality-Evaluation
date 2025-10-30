def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # While there's a pair with absolute difference > 1, insert the missing values
    while True:
        found_pair = False
        for i in range(len(A) - 1):
            if abs(A[i] - A[i+1]) > 1:
                found_pair = True
                if A[i] < A[i+1]:
                    # Insert increasing numbers between A[i] and A[i+1]
                    to_insert = list(range(A[i] + 1, A[i+1]))
                else:
                    # Insert decreasing numbers between A[i] and A[i+1]
                    to_insert = list(range(A[i] - 1, A[i+1], -1))
                
                A = A[:i+1] + to_insert + A[i+1:]
                break
        
        if not found_pair:
            break
    
    print(" ".join(map(str, A)))

# Do not remove the function call below
main()