def solve():
    import sys
    import bisect
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    D = int(input_data[2])
    
    A = list(map(int, input_data[3:3+N]))
    B = list(map(int, input_data[3+N:3+N+M]))
    
    # Sort the lists
    A.sort()
    B.sort()
    
    best_sum = -1
    
    # For each value in A, try to find a suitable value in B
    # such that |A[i] - B[j]| <= D and maximize A[i] + B[j].
    for a in A:
        # We need B[j] in [a-D, a+D]
        left = bisect.bisect_left(B, a - D)
        right = bisect.bisect_right(B, a + D) - 1
        
        if 0 <= left < len(B) and 0 <= right < len(B) and left <= right:
            # The maximum B in this valid range is B[right]
            candidate_sum = a + B[right]
            if candidate_sum > best_sum:
                best_sum = candidate_sum
    
    print(best_sum)
    
def main():
    solve()

if __name__ == "__main__":
    main()