import bisect

def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    a_sorted = sorted(a)
    test_starts = set()
    for val in a:
        test_starts.add(val - m)
        test_starts.add(val)
    
    max_gifts = 0
    for start_point in test_starts:
        l_index = bisect.bisect_left(a_sorted, start_point)
        r_prime_index = bisect.bisect_left(a_sorted, start_point + m)
        r_index = r_prime_index - 1
        current_gifts = max(0, r_index - l_index + 1)
        max_gifts = max(max_gifts, current_gifts)
        
    print(max_gifts)

if __name__ == '__main__':
    solve()