def main():
    import sys
    data = sys.stdin.read().strip().split()
    W, B = map(int, data)
    
    # The base pattern that repeats infinitely
    pattern = "wbwbwwbwbwbw"
    
    # Repeat it enough times to cover all possible substrings up to length W+B (<= 200)
    repeated_pattern = pattern * 21  # 12 * 21 = 252 characters
    
    # The length of the substring we need
    L = W + B
    
    # If the required substring length is greater than our repeated string, it's not possible
    if L > len(repeated_pattern):
        print("No")
        return
    
    # Sliding window to count how many 'w' are in each substring of length L
    current_count_w = repeated_pattern[:L].count('w')
    if current_count_w == W:
        print("Yes")
        return
    
    # Slide the window across the repeated_pattern
    for i in range(len(repeated_pattern) - L):
        if repeated_pattern[i] == 'w':
            current_count_w -= 1
        if repeated_pattern[i + L] == 'w':
            current_count_w += 1
        
        if current_count_w == W:
            print("Yes")
            return
    
    # If no suitable window found
    print("No")

# Don't forget to call main()
if __name__ == "__main__":
    main()