import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if n == 0:
        print(0)
        return
        
    M = max(A)
    freq = [0] * (M + 1)
    for a in A:
        if a <= M:
            freq[a] += 1
            
    prefix = [0] * (M + 1)
    if M >= 0:
        prefix[0] = freq[0]
        for i in range(1, M + 1):
            prefix[i] = prefix[i - 1] + freq[i]
    else:
        prefix[0] = freq[0]
        
    total = 0
    for d in range(1, M + 1):
        if freq[d] == 0:
            continue
        k = 1
        while k * d <= M:
            L = k * d
            R = min(M, (k + 1) * d - 1)
            cnt = prefix[R] - (prefix[L - 1] if L - 1 >= 0 else 0)
            total += freq[d] * cnt * k
            k += 1
            
    self_pairs = n
    same_value_pairs = 0
    for d in range(1, M + 1):
        if freq[d] >= 2:
            same_value_pairs += freq[d] * (freq[d] - 1) // 2
            
    ans = total - self_pairs - same_value_pairs
    print(ans)

if __name__ == "__main__":
    main()