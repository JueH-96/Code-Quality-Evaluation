import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    for _ in range(t):
        n = int(input[ptr])
        ptr += 1
        a = list(map(int, input[ptr:ptr+n]))
        ptr += n

        # Compute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + a[i]

        # Find all divisors of n
        divisors = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        divisors = list(divisors)

        max_diff = 0
        for k in divisors:
            m = n // k
            current_max = -float('inf')
            current_min = float('inf')
            for i in range(m):
                start = i * k
                end = start + k
                s = prefix[end] - prefix[start]
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