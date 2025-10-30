def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+M]))
    
    if M > N:
        print(-1)
        return
    
    A.sort()
    B.sort()
    
    current_pos = 0
    total = 0
    
    for i in range(M):
        b = B[i]
        low = current_pos
        high = len(A) - 1
        res = -1
        while low <= high:
            mid = (low + high) // 2
            if A[mid] >= b:
                res = mid
                high = mid - 1
            else:
                low = mid + 1
        if res == -1:
            print(-1)
            return
        # Check if there are enough elements left after selecting res
        remaining = len(A) - res - 1
        needed = M - i - 1
        if remaining < needed:
            print(-1)
            return
        total += A[res]
        current_pos = res + 1
    
    print(total)

if __name__ == "__main__":
    main()