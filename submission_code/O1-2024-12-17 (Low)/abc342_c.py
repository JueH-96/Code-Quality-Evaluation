def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = input_data[1]
    Q = int(input_data[2])
    
    # We'll read pairs (c, d) after that
    pairs = input_data[3:]
    
    # final_map[x] will hold the eventual replacement for character x (0-based: 'a' -> 0, 'z' -> 25).
    final_map = list(range(26))  # initially each character maps to itself
    
    idx = 0
    for _ in range(Q):
        c_i = pairs[idx]
        d_i = pairs[idx+1]
        idx += 2
        c_int = ord(c_i) - ord('a')
        d_int = ord(d_i) - ord('a')
        # Replace all final occurrences of c_int with d_int
        for x in range(26):
            if final_map[x] == c_int:
                final_map[x] = d_int
    
    # Build the final string according to final_map
    result_chars = []
    for ch in S:
        original_index = ord(ch) - ord('a')
        mapped_index = final_map[original_index]
        result_chars.append(chr(mapped_index + ord('a')))
    
    print("".join(result_chars))

# Do not forget to call main()!
main()