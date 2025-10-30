def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    idx = 1
    
    sumA = 0
    max_diff = -10**20  # a sufficiently small number
    
    for _ in range(N):
        A = int(data[idx]); B = int(data[idx+1])
        idx += 2
        
        sumA += A
        diff = B - A
        if diff > max_diff:
            max_diff = diff
    
    print(sumA + max_diff)

# Do not forget to call main()
if __name__ == "__main__":
    main()