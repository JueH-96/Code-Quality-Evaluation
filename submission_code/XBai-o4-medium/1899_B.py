import sys
import math

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx])
    idx += 1
    results = []
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        a = list(map(int, data[idx:idx + n]))
        idx += n
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + a[i]
        # Compute divisors of n
        divisors = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        max_diff = 0
        for k in divisors:
            m = n // k
            min_sum = float('inf')
            max_sum = -float('inf')
            for i in range(m):
                start = i * k
                end = (i + 1) * k
                current_sum = prefix[end] - prefix[start]
                if current_sum > max_sum:
                    max_sum = current_sum
                if current_sum < min_sum:
                    min_sum = current_sum
            current_diff = max_sum - min_sum
            if current_diff > max_diff:
                max_diff = current_diff
        results.append(str(max_diff))
    print('
'.join(results))

if __name__ == "__main__":
    main()