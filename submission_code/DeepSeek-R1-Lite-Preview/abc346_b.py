def main():
    import sys
    W_B = sys.stdin.read().strip().split()
    if len(W_B) < 2:
        W_B += sys.stdin.read().strip().split()
    W, B = map(int, W_B)
    
    base_pattern = "wbwbwwbwbwbw"
    len_pattern = len(base_pattern)
    
    # Calculate the number of repetitions needed
    repetitions = (W + B + len_pattern - 1) // len_pattern + 1
    concatenated_string = base_pattern * repetitions
    
    window_size = W + B
    # Ensure the concatenated string is at least window_size characters
    if len(concatenated_string) < window_size:
        concatenated_string = base_pattern * ((window_size // len_pattern) + 2)
    
    # Slide the window and check for the required counts
    for i in range(len(concatenated_string) - window_size + 1):
        substring = concatenated_string[i:i+window_size]
        count_w = substring.count('w')
        count_b = substring.count('b')
        if count_w == W and count_b == B:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()