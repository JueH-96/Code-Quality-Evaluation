def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    counts = {}
    for x in a:
        counts[x] = counts.get(x, 0) + 1
    
    for _ in range(q):
        query = list(map(int, input().split()))
        index = query[0]
        value = query[1]
        old_value = a[index-1]
        
        counts[old_value] = counts.get(old_value, 0) - 1
        if counts[old_value] == 0:
            if old_value in counts:
                del counts[old_value]
                
        a[index-1] = value
        counts[value] = counts.get(value, 0) + 1
        
        mex_val = 0
        while True:
            if counts.get(mex_val, 0) == 0:
                print(mex_val)
                break
            mex_val += 1

if __name__ == '__main__':
    solve()