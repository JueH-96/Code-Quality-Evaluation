def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return

    # Read inputs
    it = iter(input_data)
    n = int(next(it))
    k = int(next(it))
    # Read sequence A
    A = [int(next(it)) for _ in range(n)]
    
    # We remove k elements and keep n-k elements.
    # Our goal is to choose a set of n-k elements (as a subsequence, order is preserved)
    # such that the difference between the maximum and the minimum value in the kept elements is minimized.
    #
    # Notice that the order of the original sequence doesn't affect the min and max.
    # We just need to pick any n-k numbers from A.
    #
    # This implies that we want to choose n-k numbers from A with minimal (max - min).
    # A well-known fact is that if you are allowed to choose any subset (ignoring original order)
    # the best you can do is to sort A and take a window of length n-k that minimizes the range.
    #
    # Even though A was given in an order that must be preserved, any subset of elements,
    # arranged in increasing order of their original index, is valid.
    #
    # Therefore, the solution is:
    #  1. Sort A.
    #  2. For every contiguous block of length (n-k), calculate the difference (last - first).
    #  3. The answer is the minimum such difference.
    
    L = n - k
    A.sort()
    
    ans = float('inf')
    for i in range(n - L + 1):
        diff = A[i + L - 1] - A[i]
        if diff < ans:
            ans = diff
            
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()