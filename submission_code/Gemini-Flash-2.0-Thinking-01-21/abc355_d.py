def solve():
    n = int(input())
    intervals = []
    for _ in range(n):
        l, r = map(int, input().split())
        intervals.append({'l': l, 'r': r})
    
    intervals.sort(key=lambda x: x['l'])
    
    intersection_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            interval1 = intervals[i]
            interval2 = intervals[j]
            l1, r1 = interval1['l'], interval1['r']
            l2, r2 = interval2['l'], interval2['r']
            if r1 >= l2 and r2 >= l1:
                intersection_count += 1
                
    print(intersection_count)

if __name__ == '__main__':
    solve()