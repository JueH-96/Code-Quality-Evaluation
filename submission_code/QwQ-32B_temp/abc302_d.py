import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    D = int(input[idx])
    idx += 1
    
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+M]))
    idx += M
    
    A.sort()
    B.sort()
    
    max_sum = -1
    
    for a in A:
        lower = a - D
        upper = a + D
        
        l = bisect.bisect_left(B, lower)
        r = bisect.bisect_right(B, upper) - 1
        
        if l <= r:
            current_b = B[r]
            current_sum = a + current_b
            if current_sum > max_sum:
                max_sum = current_sum
    
    print(max_sum if max_sum != -1 else -1)

if __name__ == "__main__":
    main()