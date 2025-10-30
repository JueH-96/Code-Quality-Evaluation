def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:N+1]))
    B = list(map(int, input_data[N+1:2*N]))
    
    # Quick impossibility check:
    # If there is an existing box that is smaller than the smallest toy,
    # that box cannot hold any toy at all (since all toys are ≥ smallest toy),
    # and we have exactly N boxes for N toys. Hence impossible.
    if min(B) < min(A):
        print(-1)
        return
    
    # Sort toys and boxes in descending order
    A.sort(reverse=True)
    B.sort(reverse=True)
    
    # We'll greedily match the largest toy with the largest available box if possible.
    # If not possible, we "skip" placing that toy in any existing box, 
    # implying it must go to the purchased box (which we can only use once).
    skip_count = 0  # How many toys we've assigned to the purchased box
    max_skipped_size = 0  # The size of the toy(s) that we had to skip
    
    i = 0  # index for toys
    j = 0  # index for boxes
    # We have N-1 boxes in B and 1 box to purchase
    
    while i < N and j < (N - 1):
        if B[j] >= A[i]:
            # We can place toy i in box j
            i += 1
            j += 1
        else:
            # We must skip placing toy i in any existing box
            skip_count += 1
            max_skipped_size = max(max_skipped_size, A[i])
            i += 1
            if skip_count > 1:
                # We needed to skip more than one toy => not feasible
                print(-1)
                return
    
    # After this, we may have leftover toys or leftover boxes.
    leftover_toys = N - i
    
    # If the number of leftover toys plus the toys already skipped exceeds 1,
    # we cannot place them all (only one purchased box).
    if leftover_toys + skip_count > 1:
        print(-1)
        return
    
    # Cases:
    # 1) leftover_toys + skip_count == 0
    #    => We placed all toys into existing boxes, never skipped any toy.
    #       The purchased box is unused, but must be a positive integer => x=1
    #    => or we used skip once, but then leftover=0 => skip=1 => that was handled inside the loop?
    # Actually handle carefully:
    # leftover_toys + skip_count can be 0 or 1.
    
    if leftover_toys + skip_count == 0:
        # All toys placed, skip_count=0, leftover=0
        # We never had to use the purchased box, so the smallest positive integer is 1
        print(1)
        return
    
    # leftover_toys + skip_count == 1
    # Either skip=1,leftover=0 or skip=0,leftover=1
    if skip_count == 1:
        # Used the purchased box for the one skipped toy
        # so x must be at least the size of that skipped toy
        print(max_skipped_size)
    else:
        # skip=0, leftover=1 => the leftover toy is A[i]
        # we must place that leftover toy in the purchased box
        print(A[i])

# Do not forget to call main()
main()