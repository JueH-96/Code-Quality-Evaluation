def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+M]))
    
    A.sort()
    B.sort()
    
    j = 0
    total = 0
    possible = True
    
    for b in B:
        while j < len(A) and A[j] < b:
            j += 1
        if j < len(A):
            total += A[j]
            j += 1
        else:
            possible = False
            break
    
    if possible:
        print(total)
    else:
        print(-1)
        
if __name__ == "__main__":
    main()