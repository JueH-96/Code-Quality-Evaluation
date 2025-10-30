import sys
from functools import lru_cache

def main():
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline().strip())
    
    len_S = len(S)
    len_N = N.bit_length() if N != 0 else 1  # Handle N=0 case, though problem states N >=1
    
    if len_S < len_N:
        max_candidate = S.replace('?', '1')
        print(int(max_candidate, 2))
    else:
        B_str = bin(N)[2:].zfill(len_S)
        # Now find the maximum number <= B_str formed by S
        
        @lru_cache(maxsize=None)
        def dp(i, is_tight):
            if i == len_S:
                return 0
            max_val = -1
            current_b = int(B_str[i])
            
            # Determine possible choices for current bit
            if S[i] == '0':
                choices = [0]
            elif S[i] == '1':
                choices = [1]
            else:
                choices = [0, 1]
            
            for bit in choices:
                if is_tight:
                    if bit > current_b:
                        continue
                    new_tight = (bit == current_b)
                else:
                    new_tight = False
                
                contribution = bit << (len_S - i - 1)
                res = dp(i + 1, new_tight)
                if res != -1:
                    total = contribution + res
                    if total > max_val:
                        max_val = total
            return max_val if max_val != -1 else -1
        
        result = dp(0, True)
        print(result if result != -1 else -1)

if __name__ == "__main__":
    main()