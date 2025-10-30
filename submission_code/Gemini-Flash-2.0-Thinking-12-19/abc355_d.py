def solve():
    n = int(input())
    intervals = []
    for _ in range(n):
        l, r = map(int, input().split())
        intervals.append({'l': l, 'r': r})
    
    sorted_intervals = sorted(intervals, key=lambda x: x['l'])
    
    intersection_count = 0
    for i in range(n - 1):
        r_i = sorted_intervals[i]['r']
        
        max_index = -1
        low = i + 1
        high = n - 1
        
        while low <= high:
            mid = (low + high) // 2
            if sorted_intervals[mid]['l'] <= r_i:
                max_index = mid
                low = mid + 1
            else:
                high = mid - 1
                
        if max_index != -1:
            intersection_count += (max_index - i)
            
    print(intersection_count)

if __name__ == '__main__':
    solve()