import sys
import math

def get_divisors(n):
    divisors = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return sorted(divisors)

def main():
    data = sys.stdin.read().split()
    ptr = 0
    t = int(data[ptr])
    ptr += 1
    for _ in range(t):
        n = int(data[ptr])
        ptr += 1
        a = list(map(int, data[ptr:ptr+n]))
        ptr += n
        divisors = get_divisors(n)
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i-1] + a[i-1]
        max_diff = 0
        for k in divisors:
            m = n // k
            current_max = -float('inf')
            current_min = float('inf')
            for j in range(m):
                start = j * k
                end = start + k
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