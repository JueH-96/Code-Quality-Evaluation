import sys
import math

def main():
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
        
        # Compute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + a[i]
        
        # Function to get all divisors of n
        def get_divisors(n):
            divisors = set()
            for i in range(1, int(math.isqrt(n)) + 1):
                if n % i == 0:
                    divisors.add(i)
                    divisors.add(n // i)
            return divisors
        
        divisors = get_divisors(n)
        max_diff = 0
        
        for k in divisors:
            m = n // k
            # Calculate all group sums using list comprehension
            sums = [prefix[i*k + k] - prefix[i*k] for i in range(m)]
            current_max = max(sums)
            current_min = min(sums)
            diff = current_max - current_min
            if diff > max_diff:
                max_diff = diff
        
        print(max_diff)

if __name__ == "__main__":
    main()