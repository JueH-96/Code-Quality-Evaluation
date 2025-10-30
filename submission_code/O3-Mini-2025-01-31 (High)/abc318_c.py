def main():
    import sys
    
    data = sys.stdin.read().split()
    if not data:
        return
    
    # Read input values.
    N = int(data[0])
    D = int(data[1])
    P = int(data[2])
    fares = list(map(int, data[3:]))

    total = sum(fares)
    
    # Sort the fares in descending order so that if you cover k days,
    # you cover the k most expensive days.
    fares.sort(reverse=True)
    
    # Build a prefix sum array such that prefix[k] is the sum of the k largest fares.
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i + 1] = prefix[i] + fares[i]
    
    # Initialize answer with the case of using no one-day passes.
    ans = total
    
    # For each possible count k of days (0 <= k <= N) covered by one-day passes,
    # the total cost would be:
    #   cost = [total regular fares for all days] - [saving on days covered by passes]
    #          + [cost for buying enough batches for k passes].
    # Since we want maximum saving, we choose the k most expensive fares.
    # Passing a day with a pass eliminates its fare, but the cost of that pass comes from 
    # buying a batch of D one-day passes for P yen. The number of batches required is ceil(k/D),
    # so the cost for passes is ceil(k/D)*P = ((k + D - 1)//D)*P.
    
    for k in range(0, N + 1):
        batches = 0 if k == 0 else (k + D - 1) // D
        cost = total - prefix[k] + batches * P
        if cost < ans:
            ans = cost

    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()