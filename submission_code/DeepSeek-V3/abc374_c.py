# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = list(map(int, data[1:N+1]))
    
    total = sum(K)
    half = total // 2
    
    # Since N is small (up to 20), we can use bitmask to try all possible combinations
    min_max = float('inf')
    
    # Iterate over all possible subsets
    for mask in range(0, 2**N):
        sum_A = 0
        for i in range(N):
            if mask & (1 << i):
                sum_A += K[i]
        sum_B = total - sum_A
        current_max = max(sum_A, sum_B)
        if current_max < min_max:
            min_max = current_max
    
    print(min_max)

if __name__ == "__main__":
    main()