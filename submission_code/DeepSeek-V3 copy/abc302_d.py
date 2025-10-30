# YOUR CODE HERE
import bisect

def main():
    N, M, D = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
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
            max_b = B[right - 1]
            current_sum = a + max_b
            if current_sum > max_sum:
                max_sum = current_sum
    
    print(max_sum)

if __name__ == "__main__":
    main()