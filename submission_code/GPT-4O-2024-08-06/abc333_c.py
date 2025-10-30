# YOUR CODE HERE
def find_nth_repunits_sum(N):
    # Generate repunits
    repunits = []
    for k in range(1, 13):  # 12 is enough because 111111111111 is already a very large number
        repunits.append((10**k - 1) // 9)
    
    # Generate all possible sums of three repunits
    sums = set()
    for i in range(len(repunits)):
        for j in range(len(repunits)):
            for k in range(len(repunits)):
                sums.add(repunits[i] + repunits[j] + repunits[k])
    
    # Sort the sums
    sorted_sums = sorted(sums)
    
    # Return the N-th smallest sum
    return sorted_sums[N-1]

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    print(find_nth_repunits_sum(N))