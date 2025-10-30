import sys

def get_divisors(n):
    divisors = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return divisors

def main():
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    for _ in range(t):
        n = int(input[ptr])
        ptr += 1
        a = list(map(int, input[ptr:ptr + n]))
        ptr += n
        
        # Compute prefix sum
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + a[i]
        
        # Get divisors
        divisors = get_divisors(n)
        max_diff = 0
        
        for k in divisors:
            groups = n // k
            if groups == 1:
                current = 0
            else:
                min_s = float('inf')
                max_s = -float('inf')
                for i in range(groups):
                    start = i * k
                    end = start + k
                    s = prefix[end] - prefix[start]
                    if s < min_s:
                        min_s = s
                    if s > max_s:
                        max_s = s
                current = max_s - min_s
            if current > max_diff:
                max_diff = current
        
        print(max_diff)

if __name__ == "__main__":
    main()