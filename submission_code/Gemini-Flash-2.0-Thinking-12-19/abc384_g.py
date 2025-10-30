def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    k = int(input())
    queries = []
    for _ in range(k):
        queries.append(list(map(int, input().split())))
    
    results = []
    for x_k, y_k in queries:
        a_prefix = a[:x_k]
        b_prefix = b[:y_k]
        b_prefix_sorted = sorted(b_prefix)
        prefix_sum_b = [0] * (y_k + 1)
        for i in range(y_k):
            prefix_sum_b[i+1] = prefix_sum_b[i] + b_prefix_sorted[i]
            
        current_sum = 0
        for a_val in a_prefix:
            low = 0
            high = y_k - 1
            p_i = -1
            while low <= high:
                mid = (low + high) // 2
                if b_prefix_sorted[mid] <= a_val:
                    p_i = mid
                    low = mid + 1
                else:
                    high = mid - 1
            
            count_le_ai = 0
            if p_i != -1:
                count_le_ai = p_i + 1
            else:
                count_le_ai = 0
                
            term_i = (2 * count_le_ai - y_k) * a_val + prefix_sum_b[y_k] - 2 * prefix_sum_b[count_le_ai]
            current_sum += term_i
            
        results.append(current_sum)
        
    for res in results:
        print(res)

if __name__ == '__main__':
    solve()