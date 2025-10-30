def main():
    import sys
    input = sys.stdin.readline
    
    N, A, B = map(int, input().split())
    D = list(map(int, input().split()))
    M = A + B
    
    # We represent possible x mod M as a union of up to 2 intervals in [0, M-1].
    # Start with the whole circle:
    curr = [(0, M - 1)]
    
    for d in D:
        # For plan at D_i = d, acceptable x satisfy
        #   (x + d) mod M in [1..A]
        # => x mod M in [1-d .. A-d] (mod M) as a cyclic interval.
        low = (1 - d) % M
        high = (A - d) % M
        
        # Build the list of at most two intervals for this constraint
        if low <= high:
            need = [(low, high)]
        else:
            # wraps around
            need = [(low, M - 1), (0, high)]
        
        # Intersect current possibilities with these
        new_curr = []
        for (l1, r1) in curr:
            for (l2, r2) in need:
                l = max(l1, l2)
                r = min(r1, r2)
                if l <= r:
                    new_curr.append((l, r))
        curr = new_curr
        
        # If no possible x remains, we can stop
        if not curr:
            print("No")
            return
    
    # If after all plans there's still some valid x, answer is Yes
    print("Yes")

if __name__ == "__main__":
    main()