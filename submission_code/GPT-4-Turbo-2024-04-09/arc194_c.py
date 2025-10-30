def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:2*N+1]))
    C = list(map(int, data[2*N+1:3*N+1]))
    
    total_cost = 0
    
    for i in range(N):
        if A[i] != B[i]:
            total_cost += C[i]
    
    print(total_cost)

if __name__ == "__main__":
    main()