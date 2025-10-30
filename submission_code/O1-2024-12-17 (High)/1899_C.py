def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    idx = 1
    
    answers = []
    for _ in range(t):
        n = int(data[idx]); idx += 1
        arr = list(map(int, data[idx:idx+n]))
        idx += n
        
        # If there's only one element, the answer is that element
        if n == 1:
            answers.append(str(arr[0]))
            continue
        
        # dp_prev will track the best sum for a valid subarray ending at (i-1)
        dp_prev = arr[0]
        max_so_far = dp_prev
        
        # We'll track the parity of the previous element
        # (Python's %2 for negative odd gives 1, negative even gives 0, so this works fine)
        p_prev = arr[0] % 2
        
        for i in range(1, n):
            p_cur = arr[i] % 2
            if p_cur != p_prev:
                # We can extend the subarray if parities differ
                dp_cur = dp_prev + arr[i]
                # Check if starting fresh from current element is better
                if dp_cur < arr[i]:
                    dp_cur = arr[i]
            else:
                # Must start a new subarray if parities are the same
                dp_cur = arr[i]
            
            # Update the global max
            if dp_cur > max_so_far:
                max_so_far = dp_cur
            
            dp_prev = dp_cur
            p_prev = p_cur
        
        answers.append(str(max_so_far))
    
    # Print the answers for each test case
    print("
".join(answers))

# Do not forget to call main()
if __name__ == "__main__":
    main()