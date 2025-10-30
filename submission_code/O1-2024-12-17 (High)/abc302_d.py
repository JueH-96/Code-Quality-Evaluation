# YOUR CODE HERE
def main():
    import sys
    from bisect import bisect_right
    
    data = list(map(int, sys.stdin.read().strip().split()))
    N, M, D = data[0], data[1], data[2]
    idx = 3
    A = data[idx:idx+N]
    idx += N
    B = data[idx:idx+M]
    
    B.sort()
    max_sum = -1
    
    for a in A:
        left = a - D
        right = a + D
        
        # Find the index of the largest B-element <= right
        r_idx = bisect_right(B, right) - 1
        # Check if that B is also >= left
        if r_idx >= 0 and B[r_idx] >= left:
            candidate_sum = a + B[r_idx]
            if candidate_sum > max_sum:
                max_sum = candidate_sum
    
    print(max_sum)

main()