def main():
    import sys
    import bisect
    
    data = sys.stdin.read().strip().split()
    N, M, D = map(int, data[:3])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+N+M]))
    
    A.sort()
    B.sort()
    
    answer = -1
    
    for a in A:
        # Find the largest index hi where B[hi] <= a + D
        hi = bisect.bisect_right(B, a + D) - 1
        
        # Find the smallest index lo where B[lo] >= a - D
        lo = bisect.bisect_left(B, a - D)
        
        # If a valid range exists, pick the largest B in [lo, hi], which is B[hi]
        if 0 <= lo <= hi < M:
            current_sum = a + B[hi]
            if current_sum > answer:
                answer = current_sum
    
    print(answer)

# Don't forget to call main() at the end
if __name__ == "__main__":
    main()