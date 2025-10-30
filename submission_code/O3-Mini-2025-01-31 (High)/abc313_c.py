def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    # Compute total sum, and find q and r such that:
    # S = n*q + r, with 0 <= r < n.
    total = sum(A)
    q = total // n
    r = total % n
    
    # In the final configuration, since the difference between
    # the minimum and maximum should be at most 1, the only possible
    # values are q and q+1. In fact, to match the total sum,
    # exactly r numbers must be q+1 and n-r must be q.
    #
    # For each index, if we choose its final value as:
    #   • q, then if A[i] is above q, we must remove (A[i] - q) units.
    #   • q+1, then if A[i] is above q+1, we must remove A[i] - (q+1) units.
    #
    # For any index with A[i] > q, the saving of assigning it to q+1 vs q
    # is exactly 1 unit (because A[i]-q - (A[i] - (q+1)) = 1).
    # Therefore, the overall number of moves required is:
    #   sum(max(0, A[i] - q)) over all i
    # minus a saving of 1 for each index we can assign to q+1,
    # i.e. we can reduce the cost by min(r, count{ A[i] > q } ).
    
    surplus_sum = 0
    count_surplus = 0
    for a in A:
        if a > q:
            surplus_sum += a - q
            count_surplus += 1
            
    # Save one move for each index (up to r many) if chosen as q+1.
    ans = surplus_sum - min(r, count_surplus)
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()