import bisect

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    k = int(input())
    queries = []
    for _ in range(k):
        queries.append(list(map(int, input().split())))
    
    results = []
    for query in queries:
        x_k, y_k = query[0], query[1]
        a_prefix = a[:x_k]
        b_prefix = b[:y_k]
        sorted_b_prefix = sorted(b_prefix)
        prefix_sum_b = [0] * (y_k + 1)
        for i in range(y_k):
            prefix_sum_b[i+1] = prefix_sum_b[i] + sorted_b_prefix[i]
        sum_sorted_b = prefix_sum_b[y_k]
        
        current_sum = 0
        for val_a in a_prefix:
            j0 = bisect.bisect_right(sorted_b_prefix, val_a)
            e_i = (2 * j0 - y_k) * val_a + sum_sorted_b - 2 * prefix_sum_b[j0]
            current_sum += e_i
        results.append(current_sum)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()