def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Read inputs: N is the total length, K is the number removed.
    # Hence, the remaining sequence has length L = N - K.
    N = int(data[0])
    K = int(data[1])
    L = N - K
    A = list(map(int, data[2:2+N]))
    
    # We want to choose L elements such that the difference between the maximum 
    # and the minimum among them is minimized. Notice that we’re allowed to choose 
    # any subset of indices, so the ordering restriction is irrelevant for the 
    # min-max difference (any set of L elements can be arranged in increasing order 
    # according to their positions in A).
    #
    # Therefore, the strategy is: sort the values in A. In the sorted order, 
    # the tightest L numbers (i.e. consecutive in the sorted order) will yield the 
    # smallest possible difference (max-min).
    #
    # Example: A = (3,1,5,4,9) with K = 2 gives L = 3. Sorting gives (1,3,4,5,9).
    # The best consecutive block of length 3 is (3,4,5) with a difference of 5-3 = 2.
    
    A.sort()
    
    answer = float('inf')
    for i in range(N - L + 1):
        current_range = A[i + L - 1] - A[i]
        if current_range < answer:
            answer = current_range
            
    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()