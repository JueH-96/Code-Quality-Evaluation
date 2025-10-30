def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    W, B = map(int, input_data)
    
    S = "wbwbwwbwbwbw"  # length = 12
    total_length = W + B
    
    # Edge case: if total_length == 0 (but the problem states W+B >= 1, so likely unnecessary)
    if total_length == 0:
        print("No")
        return
    
    # Repeat S enough times so we can capture any wraparound opportunities
    # Enough so that there's definitely room to slide a window of length total_length
    # Repeating 3*(W+B+1) is safely above what we need for W,B <= 100
    repeat_count = 3 * (W + B + 1)
    T = S * repeat_count
    
    # Build prefix sums to quickly count 'w' in any substring
    prefix_w = [0] * (len(T) + 1)
    for i, ch in enumerate(T):
        prefix_w[i + 1] = prefix_w[i] + (1 if ch == 'w' else 0)
    
    # Slide over T to check if there's a substring of length W+B
    # containing exactly W 'w' (and thus B 'b')
    for start in range(len(T) - total_length + 1):
        w_count = prefix_w[start + total_length] - prefix_w[start]
        if w_count == W:
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()