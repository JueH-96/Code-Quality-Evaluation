def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Dictionary to store the positions of each element
    from collections import defaultdict
    positions = defaultdict(list)
    
    for index, value in enumerate(A):
        positions[value].append(index)
    
    total_count = 0
    
    # Iterate over each unique value in A
    for key, pos_list in positions.items():
        if len(pos_list) < 2:
            continue
        
        # For each position i in the list of positions for the current value
        for i in range(len(pos_list) - 1):
            # i_index is the actual index in the array A
            i_index = pos_list[i]
            
            # Count elements between i_index and the next occurrence of the same value
            for j in range(i + 1, len(pos_list)):
                k_index = pos_list[j]
                
                # We need to count the number of unique j's such that i_index < j < k_index
                # and A[j] != A[i_index]
                # This is equivalent to counting all j's minus those where A[j] == A[i_index]
                
                # Count all possible j's
                count_j = k_index - i_index - 1
                
                # Subtract the j's where A[j] == A[i_index]
                # This is the number of positions between i_index and k_index in pos_list
                # excluding the current i_index and k_index themselves
                count_same = j - i - 1
                
                # Valid j's are those which are not the same as A[i_index]
                count_valid_j = count_j - count_same
                total_count += count_valid_j
    
    print(total_count)

if __name__ == "__main__":
    main()