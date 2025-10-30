def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    idx = 1
    
    # Helper function to get parity in [0,1] for both positive/negative numbers
    parity = lambda x: (x % 2 + 2) % 2
    
    results = []
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        
        arr = list(map(int, data[idx:idx+n]))
        idx += n
        
        # dp_current will hold the maximum sum of a valid subarray ending at the current element
        dp_current = arr[0]
        max_sum = dp_current
        
        for i in range(1, n):
            if parity(arr[i]) != parity(arr[i-1]):
                # If parity differs, we can extend the subarray
                dp_current += arr[i]
            else:
                # Otherwise, start a new subarray from the current element
                dp_current = arr[i]
            
            # Keep track of the global maximum
            if dp_current > max_sum:
                max_sum = dp_current
        
        results.append(str(max_sum))
    
    print("
".join(results))

# Don't forget to call main()!
if __name__ == "__main__":
    main()