def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # Read black and white ball values.
    blacks = [int(next(it)) for _ in range(N)]
    whites = [int(next(it)) for _ in range(M)]
    
    # Sort blacks in descending order.
    blacks.sort(reverse=True)
    
    # Build prefix sums for blacks.
    # prefix_black[i] = sum of the first i balls in the sorted list.
    prefix_black = [0] * (N + 1)
    for i in range(N):
        prefix_black[i+1] = prefix_black[i] + blacks[i]
    
    # Count how many black balls are positive.
    # Note that in descending order all positives come first.
    pos_count = 0
    for b in blacks:
        if b > 0:
            pos_count += 1
        else:
            break
    
    # For whites, we only ever want to include those with a positive value.
    # (since adding a white ball forces us to also pick a black ball;
    # if a white ball is negative, it would only hurt the total sum)
    white_pos = [w for w in whites if w > 0]
    white_pos.sort(reverse=True)
    
    # Build prefix sums for white positives.
    L = len(white_pos)
    prefix_white = [0] * (L + 1)
    for i in range(L):
        prefix_white[i+1] = prefix_white[i] + white_pos[i]
    
    # We need to choose some number k of white balls such that 
    # (i) we take the k highest white positives (the best choice), and 
    # (ii) we must take at least k black balls.
    #
    # How do we optimize the black-ball contribution?
    # If we are forced to take k blacks then:
    #  • If k is not larger than the number of positive blacks, we can take
    #    all black balls that have positive value (which is the first pos_count balls).
    #    Their sum is prefix_black[pos_count].
    #  • If k > pos_count then we are forced to also take some negatives (since blacks are sorted descending)
    #    and the best k-black sum is prefix_black[k].
    #
    # We must consider k = 0 as well (i.e. choosing no white balls).
    # Finally, the overall answer is the maximum over 0 ≤ k ≤ min(L, N).
    ans = prefix_black[pos_count]  # k = 0 option: only take positive blacks.
    max_k = min(L, N)
    # Try each k >= 1 white balls chosen.
    for k in range(1, max_k+1):
        if k <= pos_count:
            black_sum = prefix_black[pos_count]
        else:
            black_sum = prefix_black[k]
        # Total sum: best black sum subject to taking at least k + sum of top k white positives.
        total = black_sum + prefix_white[k]
        if total > ans:
            ans = total
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()