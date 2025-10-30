import math

def get_divisors(n):
    divisors = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return divisors

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    for _ in range(t):
        n = int(input[ptr])
        ptr += 1
        a = list(map(int, input[ptr:ptr + n]))
        ptr += n
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + a[i]
        max_a = max(a)
        min_a = min(a)
        divisors = get_divisors(n)
        global_max = 0
        for k in divisors:
            if n % k != 0:
                continue
            m = n // k
            if m == 1:
                current_diff = 0
            elif k == 1:
                current_diff = max_a - min_a
            else:
                current_max = -float('inf')
                current_min = float('inf')
                for j in range(m):
                    start = j * k
                    end = start + k
                    sum_j = prefix[end] - prefix[start]
                    if sum_j > current_max:
                        current_max = sum_j
                    if sum_j < current_min:
                        current_min = sum_j
                current_diff = current_max - current_min
            if current_diff > global_max:
                global_max = current_diff
        print(global_max)

if __name__ == "__main__":
    main()