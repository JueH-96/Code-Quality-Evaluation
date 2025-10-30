def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # To minimize the difference between the maximum and minimum in B, we need to remove the K elements that are either the smallest or largest.
    # So, we can consider all possible ways to remove x smallest and (K - x) largest elements, where x ranges from 0 to K.
    
    # First, sort the array to easily find the smallest and largest elements.
    A_sorted = sorted(A)
    
    min_diff = float('inf')
    
    for x in range(K + 1):
        # Remove x smallest and (K - x) largest
        # The remaining elements are from x to N - (K - x) - 1
        left = x
        right = N - (K - x)
        current_diff = A_sorted[right - 1] - A_sorted[left]
        if current_diff < min_diff:
            min_diff = current_diff
    
    print(min_diff)

if __name__ == "__main__":
    main()