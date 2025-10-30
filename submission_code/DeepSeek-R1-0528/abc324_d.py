import math

def main():
    n = int(input().strip())
    s = input().strip()
    
    freq_S = [0] * 10
    for char in s:
        d = int(char)
        freq_S[d] += 1
        
    max_val = 10**n - 1
    max_root = math.isqrt(max_val)
    
    count = 0
    for x in range(0, max_root + 1):
        T = x * x
        t_str = str(T)
        L = len(t_str)
        
        freq_cand = [0] * 10
        for char in t_str:
            d = int(char)
            freq_cand[d] += 1
        freq_cand[0] += (n - L)
        
        if freq_cand == freq_S:
            count += 1
            
    print(count)

if __name__ == '__main__':
    main()