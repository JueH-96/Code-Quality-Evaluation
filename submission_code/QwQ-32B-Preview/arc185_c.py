def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Create a list of tuples (value, original index)
    indexed_A = [(A[i], i+1) for i in range(N)]
    # Sort the list based on values
    sorted_A = sorted(indexed_A, key=lambda x: x[0])
    
    for i in range(N-2):
        if i > 0 and sorted_A[i][0] == sorted_A[i-1][0]:
            continue
        j = i + 1
        k = N - 1
        while j < k:
            current_sum = sorted_A[i][0] + sorted_A[j][0] + sorted_A[k][0]
            if current_sum == X:
                # Print indices in ascending order
                indices = sorted([sorted_A[i][1], sorted_A[j][1], sorted_A[k][1]])
                print(indices[0], indices[1], indices[2])
                return
            elif current_sum < X:
                j += 1
            else:
                k -= 1
    print(-1)

if __name__ == "__main__":
    main()