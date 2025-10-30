# YOUR CODE HERE
def main():
    N = int(input())
    A = list(map(int, input().split()))
    
    # Precompute positions for each value
    positions = {}
    for i, a in enumerate(A):
        if a not in positions:
            positions[a] = []
        positions[a].append(i)
    
    count = 0
    for pos_list in positions.values():
        for idx1 in range(len(pos_list)):
            i = pos_list[idx1]
            for idx2 in range(idx1+1, len(pos_list)):
                k = pos_list[idx2]
                
                # Total positions between i and k
                total_positions = k - i - 1
                
                # Number of positions with the same value as A[i] between i and k
                same_count = idx2 - idx1 - 1
                
                # Number of positions with different values than A[i] between i and k
                diff_count = total_positions - same_count
                
                count += diff_count
    
    print(count)

if __name__ == "__main__":
    main()