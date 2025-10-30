import sys
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    # Precompute smallest prime factors (SPF) up to 200000
    max_num = 200000
    spf = list(range(max_num + 1))
    spf[0] = spf[1] = 0
    for i in range(2, int(max_num**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, max_num + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    count_zeros = 0
    square_free_counts = defaultdict(int)
    
    for num in A:
        if num == 0:
            count_zeros += 1
            continue
        
        x = num
        factors = []
        while x > 1:
            p = spf[x]
            if p == 0:
                break  # This should not happen for num > 0
            factors.append(p)
            x = x // p
        
        product = 1
        prev = -1
        cnt = 0
        for p in factors:
            if p == prev:
                cnt += 1
            else:
                if prev != -1 and cnt % 2 == 1:
                    product *= prev
                prev = p
                cnt = 1
        if prev != -1 and cnt % 2 == 1:
            product *= prev
        
        square_free_counts[product] += 1
    
    m = N - count_zeros
    case_1 = (N * (N - 1) // 2) - (m * (m - 1) // 2)
    case_2 = sum(cnt * (cnt - 1) // 2 for cnt in square_free_counts.values())
    
    print(case_1 + case_2)

if __name__ == "__main__":
    main()