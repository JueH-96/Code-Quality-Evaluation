def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))
    # A is sorted in ascending order
    count = 0
    p = 0  # p will point to the number of indices i (from 0 to j-1) 
           # that satisfy 2*A[i] <= A[j]
    for j in range(n):
        # For each candidate for the bottom mochi A[j],
        # we advance pointer p (which only increases as j increases)
        # such that all indices < p are valid top mochi for current A[j]
        while p < j and 2 * A[p] <= A[j]:
            p += 1
        # p is exactly the count of valid indices i in [0, j)
        count += p
    sys.stdout.write(str(count))
    
if __name__ == '__main__':
    main()