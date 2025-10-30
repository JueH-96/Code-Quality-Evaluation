def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    A.sort()
    B.sort()
    
    A_ptr = 0
    B_ptr = 0
    total_cost = 0
    assigned = 0
    
    while assigned < M and A_ptr < N:
        if A[A_ptr] >= B[B_ptr]:
            total_cost += A[A_ptr]
            assigned += 1
            B_ptr += 1
        A_ptr += 1
    
    if assigned == M:
        print(total_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()