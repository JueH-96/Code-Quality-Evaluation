def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    
    sum_nutrients = [0] * M
    idx = 2 + M
    for _ in range(N):
        for j in range(M):
            sum_nutrients[j] += int(data[idx])
            idx += 1
    
    for j in range(M):
        if sum_nutrients[j] < A[j]:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()