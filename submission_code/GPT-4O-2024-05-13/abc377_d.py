# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    intervals = []
    index = 2
    for _ in range(N):
        L = int(data[index])
        R = int(data[index + 1])
        intervals.append((L, R))
        index += 2
    
    # Create a list to mark invalid intervals
    invalid = [[False] * (M + 1) for _ in range(M + 1)]
    
    for L, R in intervals:
        for l in range(1, L + 1):
            for r in range(R, M + 1):
                invalid[l][r] = True
    
    count = 0
    for l in range(1, M + 1):
        for r in range(l, M + 1):
            if not invalid[l][r]:
                count += 1
    
    print(count)

if __name__ == "__main__":
    main()