import bisect

def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    queries = []
    for _ in range(q):
        b_k = list(map(int, input().split()))
        queries.append({'b': b_k[0], 'k': b_k[1]})
    
    a.sort()
    results = []
    for query in queries:
        b_j = query['b']
        k_j = query['k']
        low = 0
        high = 2 * 10**8 + 1
        ans = -1
        while low <= high:
            mid_distance = (low + high) // 2
            lower_bound = b_j - mid_distance
            upper_bound = b_j + mid_distance
            start_index = bisect.bisect_left(a, lower_bound)
            end_index = bisect.bisect_right(a, upper_bound)
            count = end_index - start_index
            if count >= k_j:
                ans = mid_distance
                high = mid_distance - 1
            else:
                low = mid_distance + 1
        results.append(ans)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()