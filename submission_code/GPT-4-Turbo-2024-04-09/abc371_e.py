def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # We will use a two-pointer technique to calculate the sum of distinct counts
    from collections import defaultdict
    
    # This will hold the count of each element in the current window
    count = defaultdict(int)
    # This will hold the total sum of f(i, j) for all i and j
    total_sum = 0
    # This will hold the number of distinct elements in the current window
    distinct_count = 0
    
    # j is the right end of the window
    j = 0
    
    # i is the left end of the window
    for i in range(N):
        # Move j to the right as far as we can while maintaining distinct elements count
        while j < N and (count[A[j]] == 0 or i == j):
            if count[A[j]] == 0:
                distinct_count += 1
            count[A[j]] += 1
            j += 1
        
        # Now [i, j-1] is the maximum range where all elements are within the window
        # starting from i and all elements are distinct
        total_sum += (j - i)
        
        # Now we need to move i to the right, so we need to update the counts
        count[A[i]] -= 1
        if count[A[i]] == 0:
            distinct_count -= 1
    
    print(total_sum)

if __name__ == "__main__":
    main()