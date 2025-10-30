import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    D = int(data[idx+2])
    idx += 3
    A = list(map(int, data[idx:idx+N]))
    idx += N
    B = list(map(int, data[idx:idx+M]))
    idx += M
    
    B.sort()
    max_sum = -1
    
    for a in A:
        low_val = a - D
        high_val = a + D
        # Find the left position in B where B[j] >= low_val
        low_pos = bisect.bisect_left(B, low_val)
        # Find the right position in B where B[j] <= high_val
        high_pos = bisect.bisect_right(B, high_val) - 1
        
        if low_pos <= high_pos and high_pos < len(B):
            current_sum = a + B[high_pos]
            if current_sum > max_sum:
                max_sum = current_sum
    
    print(max_sum if max_sum != -1 else -1)

if __name__ == "__main__":
    main()