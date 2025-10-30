def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    sorted_A = sorted(A)
    sorted_B = sorted(B)
    
    j = 0
    total_cost = 0
    for b in sorted_B:
        while j < N and sorted_A[j] < b:
            j += 1
        if j == N:
            print(-1)
            return
        total_cost += sorted_A[j]
        j += 1
    print(total_cost)

if __name__ == '__main__':
    main()