# YOUR CODE HERE
def main():
    import sys, bisect
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    try:
        N = int(data[0])
    except:
        return
    # Get the target sequence T (as a string)
    T = data[1].decode() if isinstance(data[1], bytes) else data[1]
    m = len(T)
    # Read all S strings.
    S_list = [ (data[i+2].decode() if isinstance(data[i+2], bytes) else data[i+2])
               for i in range(N) ]
    
    # For each string S, we compute:
    #    head = f(S,0) which is, scanning S from left-to-right with pointer starting at 0,
    #           how many characters of T are matched.
    #    req = the smallest p in [0, m) such that if we “simulate” S starting with pointer=p,
    #          we eventually complete T (i.e. pointer reaches m).
    #          (If S is complete on its own then we set req = 0.
    #           Otherwise if no starting p in the feasible range works we take req = INF, here INF is m+1.)
    heads = [0] * N
    reqs = [0] * N
    INF = m + 1
    # Cache T and its length
    T_str = T
    m_val = m

    # Compute head and req for each S:
    for i, s in enumerate(S_list):
        head = 0
        for ch in s:
            if head < m_val and ch == T_str[head]:
                head += 1
                if head == m_val:
                    break
        heads[i] = head
        if head == m_val:
            # S alone already “completes” T.
            reqs[i] = 0
        else:
            L = len(s)
            # To even hope that S, by matching extra characters, might complete T, we must have:
            #    m_val - p <= L   i.e.  p >= m_val - L.
            lo_bound = m_val - L if m_val - L > 0 else 0
            lo = lo_bound
            hi = m_val
            ans_req = INF
            # Binary search over p in [lo, hi)
            while lo < hi:
                mid = (lo + hi) >> 1
                pointer = mid
                for ch in s:
                    if pointer < m_val and ch == T_str[pointer]:
                        pointer += 1
                        if pointer == m_val:
                            break
                if pointer == m_val:
                    ans_req = mid
                    hi = mid
                else:
                    lo = mid + 1
            reqs[i] = ans_req

    # Now count valid pairs.
    # For S_i complete (head==m) all S_j work. 
    # For non–complete S_i with head=k, valid j are those with req(S_j) <= k.
    complete_count = 0
    for head in heads:
        if head == m_val:
            complete_count += 1

    # Build a sorted list of req values.
    sorted_reqs = sorted(reqs)
    tot = complete_count * N  # (i such that S_i is complete contribute for every j)
    for head in heads:
        if head < m_val:  # non–complete
            tot += bisect.bisect_right(sorted_reqs, head)
    sys.stdout.write(str(tot))
    
if __name__ == '__main__':
    main()