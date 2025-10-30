import math
import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    N = int(data[0].strip())
    S = data[1].strip()
    
    freq_S = [0] * 10
    for char in S:
        d = int(char)
        freq_S[d] += 1
        
    total_nonzero_S = sum(freq_S[1:])
    
    max_val = (10 ** N) - 1
    max_root = math.isqrt(max_val)
    
    count = 0
    for k in range(0, max_root + 1):
        x = k * k
        if x == 0:
            n_digits = 1
            zeros_in_x = 1
            non_zero_count = 0
            non_zero_freq = [0] * 10
        else:
            n_digits = 0
            zeros_in_x = 0
            non_zero_freq = [0] * 10
            temp = x
            while temp:
                d = temp % 10
                if d == 0:
                    zeros_in_x += 1
                else:
                    non_zero_freq[d] += 1
                n_digits += 1
                temp //= 10
            non_zero_count = n_digits - zeros_in_x
        
        total_zeros = zeros_in_x + (N - n_digits)
        if total_zeros != freq_S[0]:
            continue
            
        if non_zero_count != total_nonzero_S:
            continue
            
        valid = True
        for d in range(1, 10):
            if non_zero_freq[d] != freq_S[d]:
                valid = False
                break
                
        if valid:
            count += 1
            
    print(count)

if __name__ == "__main__":
    main()