def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Create a list of (value, index)
    indexed_A = [(A[i], i) for i in range(N)]
    # Sort the list based on values
    sorted_A = sorted(indexed_A, key=lambda x: x[0])
    
    min_diff = float('inf')
    i = 0
    prev_index = -1
    for j in range(N):
        while j - i + 1 > N - K:
            if sorted_A[i][1] >= prev_index:
                prev_index = sorted_A[i][1]
            i += 1
        if j - i + 1 == N - K:
            if sorted_A[j][1] >= prev_index:
                current_diff = sorted_A[j][0] - sorted_A[i][0]
                if current_diff < min_diff:
                    min_diff = current_diff
                prev_index = sorted_A[j][1]
    
    print(min_diff)

if __name__ == '__main__':
    main()