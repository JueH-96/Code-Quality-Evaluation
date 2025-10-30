def find_triplet():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    X = int(input[1])
    A = list(map(int, input[2:2+N]))
    
    pair_sums = {}
    first_occurrence = {}
    
    for k in range(N):
        current_value = A[k]
        target_sum = X - current_value
        
        if target_sum in pair_sums:
            i, j = pair_sums[target_sum]
            if j < k:
                print(i + 1, j + 1, k + 1)
                return
        
        # Update pair_sums with all possible pairs formed with current_value
        for v in list(first_occurrence.keys()):
            s = v + current_value
            if s not in pair_sums:
                pair_sums[s] = (first_occurrence[v], k)
            else:
                existing_i, existing_j = pair_sums[s]
                if k < existing_j:
                    pair_sums[s] = (first_occurrence[v], k)
        
        # Update first_occurrence with current_value's earliest index
        if current_value not in first_occurrence or k < first_occurrence[current_value]:
            first_occurrence[current_value] = k
    
    print(-1)

find_triplet()