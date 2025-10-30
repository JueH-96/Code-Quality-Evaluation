def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    S = data[1].strip()
    
    # We want to compute:
    #   answer = sum_{i=0}^{n-1} sum_{j=i}^{n-1} f(i,j)
    # where f(i,j) is defined by:
    #   f(i,i) = A[i]  (with A[i] = int(S[i]))
    #   for j > i, let f(i,j) = f(i,j-1) NAND A[j],
    # and NAND is defined by:
    #   x NAND y = 0 if (x,y) == (1,1), and 1 otherwise.
    #
    # To understand the recurrence note that the recurrence has only two states.
    # In particular, if you “simulate” the formula starting from index i:
    #
    #   state = A[i]
    #   for k = i+1,..,j:
    #       if state == 0, then regardless of A[k] we have
    #           state = 1   (because 0 NAND 0 = 1 and 0 NAND 1 = 1)
    #       if state == 1, then:
    #           if A[k] == '1': state becomes 0 (since 1 NAND 1 = 0)
    #           if A[k] == '0': state remains 1 (since 1 NAND 0 = 1)
    #
    # Thus, apart from the starting term f(i,i)=A[i], all other terms have the same behavior:
    #   - If the current state is 0, the step “resets” the state to 1.
    #   - Only when in state 1 and the new element is '1' do we get f(i,j)=0.
    #
    # Notice that if we consider a subarray [i,j] (with j>=i):
    #   • For j=i, f(i,i)= A[i] (which is 1 when S[i]=='1' and 0 when S[i]=='0').
    #   • For j > i, no matter what happened earlier, the very first time we process
    #     an element when the state is 1, if that element is '1', the output for that subarray becomes 0.
    #     Otherwise, the output is 1.
    #
    # Hence it is equivalent to say:
    #   For a fixed starting index i:
    #     - If i is the last index, the only subarray is f(i,i)=A[i].
    #     - Otherwise, for all subarrays starting at i, the answer is 1 for every ending index
    #       except that at the first occurrence (if any) of a '1' in positions > i (when state is 1)
    #       we get an output of 0.
    #
    # Let L = (n - i) be the number of subarrays starting at index i (i.e. j = i, i+1, ..., n-1).
    # If S[i]=='1' then:
    #    - The subarray [i,i] contributes 1.
    #    - For subarrays j>i, if there is at least one '1' in S[i+1:n],
    #         then the very first such j gives f(i,j)=0 and all others give 1.
    #         So sum = 1 + [(L-1) subarrays minus 1 “zero event”] = L - 1.
    #    - If there is no '1' in S[i+1:n], then all subarrays yield 1 and sum = L.
    #
    # If S[i]=='0' then:
    #    - [i,i] contributes 0.
    #    - For subarrays with j>i:
    #         Start: after the first step, state becomes 1 unconditionally.
    #         Then as before, if there is a '1' in S[i+1:n], the first occurrence gives a 0.
    #         So sum = (L-1) subarrays, but subtract 1 if a later '1' exists: sum = (L-1) - 1.
    #         If no '1' exists in S[i+1:n], then all contribute 1: sum = (L-1).
    #
    # We will precompute a suffix boolean array has_one such that for each i, 
    # has_one[i] is True if there is a '1' in S[i:].
    #
    # Note that when i == n-1 (i.e. L=1), the subarray [i,i] contributes A[i].
    
    n_int = n
    has_one = [False]*(n_int + 1)
    has_one[n_int] = False
    for i in range(n_int - 1, -1, -1):
        if S[i] == '1':
            has_one[i] = True
        else:
            has_one[i] = has_one[i+1]
    
    total = 0
    for i in range(n_int):
        L = n_int - i   # number of subarrays starting at i
        if L == 1:
            total += 1 if S[i] == '1' else 0
        else:
            if S[i] == '1':
                # For subarray starting at i, we already got a contribution of 1 from [i,i].
                if has_one[i+1]:
                    total += L - 1  # one of the remaining (L-1) subarrays gives 0
                else:
                    total += L
            else:  # S[i] == '0'
                # f(i,i) = 0 and (L-1) remaining
                if has_one[i+1]:
                    total += (L - 1) - 1
                else:
                    total += (L - 1)
    
    sys.stdout.write(str(total))

if __name__ == '__main__':
    main()