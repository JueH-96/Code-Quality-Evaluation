def solve():
    n, m = map(int, input().split())
    intervals = []
    for _ in range(n):
        l_i, r_i = map(int, input().split())
        intervals.append({'L': l_i, 'R': r_i})
    
    def is_contained(l, r, interval):
        return l <= interval['L'] and interval['R'] <= r
        
    def does_not_contain_any(l, r):
        for interval in intervals:
            if is_contained(l, r, interval):
                return False
        return True
        
    count = 0
    for l in range(1, m + 1):
        for r in range(l, m + 1):
            if does_not_contain_any(l, r):
                count += 1
                
    print(count)

if __name__ == '__main__':
    solve()