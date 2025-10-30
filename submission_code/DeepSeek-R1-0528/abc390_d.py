import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    total_mask = 1 << n
    sums = [0] * total_mask
    for mask in range(total_mask):
        total_val = 0
        for j in range(n):
            if mask & (1 << j):
                total_val += A[j]
        sums[mask] = total_val

    dp = [set() for _ in range(total_mask)]
    dp[0].add(0)
    
    if n >= 1:
        for mask in range(1, total_mask):
            low_bit = mask & -mask
            v = low_bit.bit_length() - 1
            rest = mask ^ (1 << v)
            
            if rest == 0:
                T = (1 << v)
                sT = sums[T]
                for val in dp[0]:
                    candidate = sT ^ val
                    dp[mask].add(candidate)
            else:
                sub = rest
                while True:
                    T = (1 << v) | sub
                    rest_mask = mask ^ T
                    sT = sums[T]
                    for val in dp[rest_mask]:
                        candidate = sT ^ val
                        dp[mask].add(candidate)
                    if sub == 0:
                        break
                    sub = (sub - 1) & rest

    answer = len(dp[total_mask-1])
    print(answer)

if __name__ == '__main__':
    main()