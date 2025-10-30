def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    if all(a == 1 for a in A):
        print("Yes")
        return
    
    covered = [False] * N
    i = 0
    while i < N:
        if A[i] == 0 and not covered[i]:
            if i + 1 >= N:
                # Try to handle circular case
                if not covered[i] and not covered[0]:
                    covered[i] = True
                    covered[0] = True
                    i += 1
                else:
                    print("No")
                    return
            else:
                covered[i] = True
                covered[i+1] = True
                i += 1
        else:
            i += 1
    
    for j in range(N):
        if A[j] == 0 and not covered[j]:
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()