def main():
    import sys
    import bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    K = int(next(it))
    queries = []
    xs = set()  # distinct prefix lengths for A we need
    ys = set()  # distinct prefix lengths for B we need
    for _ in range(K):
        X = int(next(it))
        Y = int(next(it))
        queries.append((X, Y))
        xs.add(X)
        ys.add(Y)
    
    # For each distinct X we want to cache sorted(A[:X]) and also its prefix–sums.
    sorted_A_cache = {}  # maps X -> (sorted_list, prefix_sum_list)
    for x in xs:
        subA = A[:x]
        sorted_subA = sorted(subA)
        ps = [0] * len(sorted_subA)
        s = 0
        for i, val in enumerate(sorted_subA):
            s += val
            ps[i] = s
        sorted_A_cache[x] = (sorted_subA, ps)
    
    # Similarly for B
    sorted_B_cache = {}
    for y in ys:
        subB = B[:y]
        sorted_subB = sorted(subB)
        ps = [0] * len(sorted_subB)
        s = 0
        for i, val in enumerate(sorted_subB):
            s += val
            ps[i] = s
        sorted_B_cache[y] = (sorted_subB, ps)
    
    # To compute the cross–sum S = ∑|u - v| between two sorted lists U and V,
    # we use the identity below.
    # (When iterating over U, for each u in U use:)
    #   contribution = u * (2*pos - len(V)) + (totalV - 2*(prefix sum on V for first pos elements))
    # where pos = bisect.bisect_right(V, u)
    #
    # In our code we choose to iterate over the smaller list.
    
    out_lines = []
    for (X, Y) in queries:
        # We decide which prefix is smaller:
        # – if X <= Y, then we use sorted(A[:X]) as the “small” list
        #   and call sorted(B[:Y]) our “large” list.
        # – Else we do the opposite.
        if X <= Y:
            small_list, _ = sorted_A_cache[X]
            large_list, large_ps = sorted_B_cache[Y]
            n_large = len(large_list)   # (this is Y)
        else:
            small_list, _ = sorted_B_cache[Y]
            large_list, large_ps = sorted_A_cache[X]
            n_large = len(large_list)   # (this is X)
        total = 0
        tot_large = large_ps[-1]  # total sum of the larger list
        for x in small_list:
            pos = bisect.bisect_right(large_list, x)
            # For an element x (from the "small" sorted list) the contribution is:
            #    x*(2*pos - n_large) + ( tot_large - 2*(prefix sum up to pos) )
            part = x * (2 * pos - n_large)
            if pos > 0:
                part += tot_large - 2 * large_ps[pos - 1]
            else:
                part += tot_large
            total += part
        out_lines.append(str(total))
    
    sys.stdout.write("
".join(out_lines))
 
if __name__ == '__main__':
    main()