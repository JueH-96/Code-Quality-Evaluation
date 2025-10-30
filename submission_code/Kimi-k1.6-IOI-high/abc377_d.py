def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx +=2
    intervals = []
    for _ in range(N):
        L = int(data[idx])
        R = int(data[idx+1])
        intervals.append((L, R))
        idx +=2
    
    # Sort intervals by their R value
    intervals.sort(key=lambda x: x[1])
    
    current_max_L = 0
    total = 0
    i = 0
    
    for r in range(1, M+1):
        # Process all intervals where R <= current r
        while i < N and intervals[i][1] <= r:
            current_max_L = max(current_max_L, intervals[i][0])
            i +=1
        # Calculate valid l's for this r
        total += max(0, r - current_max_L)
    
    print(total)

if __name__ == "__main__":
    main()