def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    
    N = int(next(it))
    schedules = []
    for _ in range(N):
        q = int(next(it))
        r = int(next(it))
        schedules.append((q, r))
    
    Q = int(next(it))
    res = []
    for _ in range(Q):
        t = int(next(it))
        d = int(next(it))
        q, r = schedules[t - 1]
        mod = d % q
        # if d is collected then answer is d
        if mod == r:
            res.append(d)
        else:
            # If remainder is less than r then simply add difference, 
            # else add a full cycle plus difference.
            if mod < r:
                res.append(d - mod + r)
            else:
                res.append(d - mod + q + r)
    
    sys.stdout.write("
".join(map(str, res)))
    
if __name__ == '__main__':
    main()