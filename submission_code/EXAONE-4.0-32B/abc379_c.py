import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(-1)
        return
    
    N = int(data[0])
    M = int(data[1])
    
    X = list(map(int, data[2:2+M]))
    A = list(map(int, data[2+M:2+2*M]))
    
    total_stones = sum(A)
    if total_stones != N:
        print(-1)
        return
        
    arr = list(zip(X, A))
    arr.sort()
    positions = [a[0] for a in arr]
    counts = [a[1] for a in arr]
    
    if positions[0] != 1:
        print(-1)
        return
        
    surplus = counts[0] - 1
    for i in range(1, len(positions)):
        gap = positions[i] - positions[i-1] - 1
        if surplus < gap:
            print(-1)
            return
        surplus -= gap
        surplus += counts[i] - 1
        
    last_gap = N - positions[-1]
    if surplus < last_gap:
        print(-1)
        return
        
    total_start_sum = 0
    for i in range(len(positions)):
        total_start_sum += positions[i] * counts[i]
        
    total_target_sum = N * (N + 1) // 2
    ans = total_target_sum - total_start_sum
    print(ans)

if __name__ == "__main__":
    main()