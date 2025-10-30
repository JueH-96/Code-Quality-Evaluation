def main():
    import sys, bisect
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    given_X1 = int(next(it))
    # We'll store trains as tuples: (S, id, A, B, T)
    # train id is 1-indexed (in input order);
    # note that train 1’s delay X[1] is fixed (given_X1).
    trains = [None]*M
    for i in range(M):
        A = int(next(it))
        B = int(next(it))
        S = int(next(it))
        T = int(next(it))
        trains[i] = (S, i+1, A, B, T)
    
    # For our BIT we wish to support queries: for a given city c and value s, quickly get 
    # max { value } among events in city c with key T <= s.
    # For every train with destination B = c, an "event" is inserted with key T and value = d + (T - S)
    # where d is computed later. We know the T–values from input. So first, group T–values by city.
    
    city_keys = {}
    for (S, tid, A, B, T) in trains:
        if B not in city_keys:
            city_keys[B] = []
        city_keys[B].append(T)
    # For each city, sort and remove duplicate keys.
    for c in city_keys:
        arr = city_keys[c]
        arr.sort()
        uniq = []
        last = None
        for v in arr:
            if v != last:
                uniq.append(v)
                last = v
        city_keys[c] = uniq

    # Build BIT (Fenwick tree) per city that appears as a destination.
    INF_NEG = -10**18
    bit_data = {}
    # For each city c, we'll store (sorted_keys, bit_array)
    for c, keys in city_keys.items():
        size = len(keys)
        bit_data[c] = (keys, [INF_NEG] * (size+1))
        
    # Define BIT query: given a city and a value s, return max value among keys <= s.
    def bit_query(city, val):
        if city not in bit_data:
            return INF_NEG
        keys, tree = bit_data[city]
        pos = bisect.bisect_right(keys, val)
        if pos == 0:
            return INF_NEG
        res = INF_NEG
        while pos:
            if tree[pos] > res:
                res = tree[pos]
            pos -= pos & -pos
        return res

    # BIT update: update at key t_val with new value new_val.
    def bit_update(city, t_val, new_val):
        if city not in bit_data:
            return
        keys, tree = bit_data[city]
        pos = bisect.bisect_left(keys, t_val) + 1
        n = len(tree) - 1
        while pos <= n:
            if new_val > tree[pos]:
                tree[pos] = new_val
            pos += pos & -pos

    # Now sort trains by scheduled departure time S (if equal, by id).
    trains.sort(key=lambda x: (x[0], x[1]))
    # delays[tid] will hold X for train tid. (X[1]=given_X1; we compute for the rest.)
    delays = [0]*(M+1)
    # We also compute d = S + X for each train.
    # For train 1, d = S + given_X1.
    # For any other train j, we set:
    #      d[j] = max( S[j],   max{ d[i] + (T[i]-S[i]) for any train i with 
    #                                     B[i] = A[j] and T[i] <= S[j] } )
    # Then X[j] = d[j] - S[j].
    for (S, tid, A, B, T) in trains:
        if tid == 1:
            d = S + given_X1
        else:
            cand = bit_query(A, S)
            if cand < S:
                d = S
            else:
                d = cand
        delays[tid] = d - S
        # Now update BIT for the destination city B with our train’s arrival event.
        # Our event key is T and value is d + (T - S) = T + X.
        event_val = d + (T - S)
        bit_update(B, T, event_val)
    
    # Print out delays for trains 2,...,M (in input order).
    # (Train 1’s delay is fixed so we do not print it.)
    out = []
    for i in range(2, M+1):
        out.append(str(delays[i]))
    sys.stdout.write(" ".join(out))
    
if __name__ == '__main__':
    main()