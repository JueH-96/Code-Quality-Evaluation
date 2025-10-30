def main():
    import sys

    data = sys.stdin.read().split()
    if not data:
        return

    n = int(data[0])
    A = list(map(int, data[1:]))
    
    # To efficiently compute the sum of contributions:
    # For each position j (1-indexed), we want to add:
    #   contribution = A_j * (number of previous elements < A_j) - (sum of previous elements < A_j)
    # We then use two Fenwick Trees (one for counts, one for sums) and process A from left to right.
    
    # Since A_i can be up to 10^8, we perform coordinate compression.
    unique_vals = sorted(set(A))
    comp = {v: i+1 for i, v in enumerate(unique_vals)}
    m = len(unique_vals)
    
    # Fenwick Tree (Binary Indexed Tree) initializations:
    BIT_count = [0] * (m + 1)
    BIT_sum = [0] * (m + 1)
    
    def update(BIT, index, val):
        while index <= m:
            BIT[index] += val
            index += index & -index
            
    def query(BIT, index):
        result = 0
        while index:
            result += BIT[index]
            index -= index & -index
        return result

    total = 0
    # Process each element in A.
    for value in A:
        idx = comp[value]
        # Query count and sum for all indices that correspond to values < current value.
        count_less = query(BIT_count, idx - 1)
        sum_less = query(BIT_sum, idx - 1)
        
        # Contribution to the result: for this element A_j, all previous A_i with A_i < A_j
        # add (A_j - A_i) which translates to A_j * count_less - sum_less.
        total += value * count_less - sum_less
        
        # Update BITs with the current value.
        update(BIT_count, idx, 1)
        update(BIT_sum, idx, value)
    
    sys.stdout.write(str(total))


if __name__ == '__main__':
    main()