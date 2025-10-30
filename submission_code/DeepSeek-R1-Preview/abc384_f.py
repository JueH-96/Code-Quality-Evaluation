def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    max_k = 30  # Sufficiently large to cover exponents up to 2^30
    
    sum_os = [0] * (max_k + 1)
    sum_2_neg = [0.0] * (max_k + 1)
    sum_o2 = [0] * (max_k + 1)
    count = [0] * (max_k + 1)
    
    prefix_sum_os = [0] * (max_k + 1)
    prefix_sum_2_neg = [0.0] * (max_k + 1)
    suffix_count = [0] * (max_k + 2)
    suffix_sum_o2 = [0] * (max_k + 2)
    
    total = 0
    
    for a in A:
        x = a
        k = 0
        while x % 2 == 0:
            x //= 2
            k += 1
        o = x
        
        sum_os_le_kj = prefix_sum_os[k]
        sum_2_neg_le_kj = prefix_sum_2_neg[k]
        part1 = sum_os_le_kj + o * (2 ** k) * sum_2_neg_le_kj
        
        count_gt_kj = suffix_count[k]
        sum_o2_gt = suffix_sum_o2[k]
        part2 = o * count_gt_kj + (sum_o2_gt // (2 ** k))  # Use integer division
        
        S_j = part1 + part2 + o
        total += S_j
        
        # Update the data structures
        sum_os[k] += o
        sum_2_neg[k] += 1.0 / (2 ** k)
        sum_o2[k] += o * (2 ** k)
        count[k] += 1
        
        # Recompute prefix sums
        current_os = 0
        current_2_neg = 0.0
        for t in range(max_k + 1):
            current_os += sum_os[t]
            prefix_sum_os[t] = current_os
            current_2_neg += sum_2_neg[t]
            prefix_sum_2_neg[t] = current_2_neg
        
        # Recompute suffix sums
        current_count = 0
        current_o2 = 0
        for t in range(max_k, -1, -1):
            current_count += count[t]
            suffix_count[t] = current_count
            current_o2 += sum_o2[t]
            suffix_sum_o2[t] = current_o2
    
    print(int(total))
    
if __name__ == '__main__':
    main()