import math

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx])
    idx += 1
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        a = list(map(int, data[idx:idx+n]))
        idx += n
        
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + a[i]
        
        divisors = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        
        max_diff = 0
        for k in divisors:
            m = n // k
            current_max = -float('inf')
            current_min = float('inf')
            for i in range(m):
                start = i * k
                end = (i + 1) * k
                s = prefix_sum[end] - prefix_sum[start]
                if s > current_max:
                    current_max = s
                if s < current_min:
                    current_min = s
            diff = current_max - current_min
            if diff > max_diff:
                max_diff = diff
        print(max_diff)

if __name__ == "__main__":
    main()