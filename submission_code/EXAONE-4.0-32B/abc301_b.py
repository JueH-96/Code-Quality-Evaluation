def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    changed = True
    while changed:
        changed = False
        i = 0
        n_curr = len(A)
        while i < n_curr - 1:
            if abs(A[i] - A[i+1]) != 1:
                changed = True
                if A[i] < A[i+1]:
                    to_insert = list(range(A[i] + 1, A[i+1]))
                else:
                    to_insert = list(range(A[i] - 1, A[i+1], -1))
                A = A[:i+1] + to_insert + A[i+1:]
                break
            else:
                i += 1
                
    print(" ".join(map(str, A)))

if __name__ == "__main__":
    main()