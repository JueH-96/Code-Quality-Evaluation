# YOUR CODE HERE
import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    D = int(data[2])
    
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+N+M]))
    
    A.sort()
    B.sort()
    
    max_sum = -1
    
    for a in A:
        # Find the range in B where b is between a - D and a + D
        low = a - D
        high = a + D
        
        # Find the leftmost index where B[i] >= low
        left = bisect.bisect_left(B, low)
        # Find the rightmost index where B[i] <= high
        right = bisect.bisect_right(B, high)
        
        if left < right:
            # There are elements in B that satisfy the condition
            # Find the maximum b in this range
            max_b = B[right-1]
            current_sum = a + max_b
            if current_sum > max_sum:
                max_sum = current_sum
    
    print(max_sum)

if __name__ == "__main__":
    main()