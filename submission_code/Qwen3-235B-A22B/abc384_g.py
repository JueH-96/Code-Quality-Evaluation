import sys
from itertools import accumulate

def main():
    input = sys.stdin.read().split()
    ptr = 0
    
    N = int(input[ptr])
    ptr += 1
    
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    K = int(input[ptr])
    ptr += 1
    
    queries = []
    for _ in range(K):
        x = int(input[ptr])
        y = int(input[ptr+1])
        queries.append((x, y))
        ptr += 2
    
    for x, y in queries:
        a_part = A[:x]
        b_part = B[:y]
        
        a_sorted = sorted(a_part)
        b_sorted = sorted(b_part)
        
        # Compute prefix sums for B_sorted
        prefix_sums = [0]
        current_sum = 0
        for num in b_sorted:
            current_sum += num
            prefix_sums.append(current_sum)
        sum_b_total = prefix_sums[-1]
        
        total = 0
        j = 0
        len_b = len(b_sorted)
        
        for a in a_sorted:
            # Move j to the first element in b_sorted greater than a
            while j < len_b and b_sorted[j] <= a:
                j += 1
            cnt_less = j
            sum_less = prefix_sums[cnt_less]
            
            contribution = a * (2 * cnt_less - y) + (sum_b_total - 2 * sum_less)
            total += contribution
        
        print(total)

if __name__ == "__main__":
    main()