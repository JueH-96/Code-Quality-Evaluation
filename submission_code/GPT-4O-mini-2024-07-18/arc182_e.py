def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    C = int(data[2])
    K = int(data[3])
    A = list(map(int, data[4:4 + N]))
    
    # Calculate the minimum values for each k
    total_sum = 0
    min_A = min(A)
    
    for k in range(K):
        current_min = (C * k + min_A) % M
        total_sum += current_min
    
    print(total_sum)

if __name__ == "__main__":
    main()