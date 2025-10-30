def main():
    import sys
    import math
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    
    # Frequency array of digits in S
    S_freq = [0]*10
    for c in S:
        S_freq[ord(c) - ord('0')] += 1
    
    # Sum of digits mod 9
    S_mod9 = sum((ord(c) - ord('0')) for c in S) % 9
    
    # The maximum integer we can form by sorting digits in descending order
    max_num = int("".join(sorted(S, reverse=True)))
    
    # Upper bound for the square root
    limit = int(math.isqrt(max_num))  # largest i so that i^2 <= max_num
    
    # We'll store valid squares we've found
    squares_found = set()
    
    # Precompute the length of S; squares with more than this many digits can stop
    length_S = len(S)
    
    for i in range(limit+1):
        # Check mod 9 quickly to skip most squares
        if ((i % 9) * (i % 9)) % 9 != S_mod9:
            continue
        
        sq = i * i
        
        # If square is bigger than max_num, we can stop
        if sq > max_num:
            break
        
        # Check digit length
        sq_str = str(sq)
        if len(sq_str) > length_S:
            break
        
        # Zero-pad to match length if needed
        sq_str = sq_str.zfill(length_S)
        
        # Build frequency of the square's digits
        freq_sq = [0]*10
        for ch in sq_str:
            freq_sq[ord(ch) - ord('0')] += 1
        
        # Compare with S's frequency
        if freq_sq == S_freq:
            squares_found.add(sq)
    
    # Print the number of distinct squares formed
    print(len(squares_found))

# Do not forget to call main() at the end
if __name__ == "__main__":
    main()