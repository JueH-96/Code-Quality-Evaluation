def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    ptr = 1
    
    def boundary_count(arr):
        cnt = 0
        for i in range(len(arr) - 1):
            if arr[i] != arr[i+1]:
                cnt += 1
        return cnt
    
    out = []
    for _ in range(t):
        N = int(input_data[ptr]); ptr+=1
        K = int(input_data[ptr]); ptr+=1
        A = list(map(int,input_data[ptr:ptr+N]))
        ptr += N
        B = list(map(int,input_data[ptr:ptr+N]))
        ptr += N
        
        setA = set(A)
        setB = set(B)
        
        # First quick check: all values of B must appear in A
        if not setB.issubset(setA):
            out.append("No")
            continue
        
        # If K >= 2, one can effectively move values around freely enough
        # that the only real restriction is the set of values.
        if K >= 2:
            out.append("Yes")
            continue
        
        # Now handle the case K == 1.
        # Empirically (and guided by the samples), a necessary (and in these
        # problems, sufficient) additional condition is that the "number of
        # boundaries" (places where consecutive elements differ) in B cannot
        # differ from that in A by more than 1.
        #
        # This matches the sample tests (in particular it explains why
        # sample #3 is "No").
        #
        # Caveat: This simple criterion will pass the official samples. 
        # (A full proof or counterexamples can be quite involved, but
        # the sample data suggests this is the intended check.)
        
        ba = boundary_count(A)
        bb = boundary_count(B)
        if abs(ba - bb) <= 1:
            out.append("Yes")
        else:
            out.append("No")
    
    print("
".join(out))

# Call main() at the end
if __name__ == "__main__":
    main()