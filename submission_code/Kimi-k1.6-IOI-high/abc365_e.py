def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] ^ a[i]
    
    sum_a = sum(a)
    
    s1 = 0
    for bit in range(31):  # considering up to 30th bit (since 1e8 is around 2^26)
        mask = 1 << bit
        cnt = 0
        for num in prefix:
            if num & mask:
                cnt += 1
        s1 += cnt * (len(prefix) - cnt) * (1 << bit)
    
    print(s1 - sum_a)

if __name__ == "__main__":
    main()