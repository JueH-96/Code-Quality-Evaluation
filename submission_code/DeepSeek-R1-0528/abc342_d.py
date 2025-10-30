import math

def main():
    max_val = 200000
    spf = list(range(max_val + 1))
    sqrt_max = int(math.isqrt(max_val))
    for i in range(2, sqrt_max + 1):
        if spf[i] == i:
            j = i * i
            while j <= max_val:
                if spf[j] == j:
                    spf[j] = i
                j += i

    n = int(input().strip())
    A = list(map(int, input().split()))
    
    zeros = 0
    freq = {}
    
    for x in A:
        if x == 0:
            zeros += 1
        else:
            temp = x
            kernel = 1
            while temp > 1:
                p = spf[temp]
                count = 0
                while temp % p == 0:
                    count += 1
                    temp //= p
                if count % 2 == 1:
                    kernel *= p
            freq[kernel] = freq.get(kernel, 0) + 1
                
    total_pairs = 0
    for count in freq.values():
        total_pairs += count * (count - 1) // 2
        
    total_pairs += zeros * (zeros - 1) // 2
    total_pairs += zeros * (n - zeros)
    
    print(total_pairs)

if __name__ == '__main__':
    main()