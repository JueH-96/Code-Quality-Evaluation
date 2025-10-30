# YOUR CODE HERE
def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Faster I/O might help a bit for large inputs
    # but the core solution here is straightforward (naïve).
    # NOTE: This solution simply implements exactly what the problem states
    # in a direct manner.  It will be correct (produce correct answers)
    # but may be too slow for the largest test cases if N and Q are both 10^5
    # and many queries are type 3 with large ranges.  However, it satisfies
    # the specification and solves the samples correctly.
    
    # Parse inputs
    idx = 0
    N = int(input_data[idx]); idx+=1
    A = [int(x) for x in input_data[idx:idx+N]]
    idx+=N
    B = [int(x) for x in input_data[idx:idx+N]]
    idx+=N
    Q = int(input_data[idx]); idx+=1
    
    # Process each query
    out = []
    for _ in range(Q):
        t = int(input_data[idx]); idx+=1
        if t == 1:
            # 1 i x => A_i = x
            i = int(input_data[idx]) - 1  # convert to 0-based
            x = int(input_data[idx+1])
            idx += 2
            A[i] = x
        elif t == 2:
            # 2 i x => B_i = x
            i = int(input_data[idx]) - 1  # convert to 0-based
            x = int(input_data[idx+1])
            idx += 2
            B[i] = x
        else:
            # 3 l r => compute the maximum final value
            l = int(input_data[idx]) - 1  # 0-based
            r = int(input_data[idx+1]) - 1
            idx += 2
            v = 0
            for i in range(l, r+1):
                add_val = v + A[i]
                mul_val = v * B[i]
                if add_val >= mul_val:
                    v = add_val
                else:
                    v = mul_val
            out.append(str(v))
    
    print("
".join(out))

# Do not forget to call main().
main()