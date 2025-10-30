import sys, math

def main():
    import sys
    import sys
    from sys import stdin
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx]); idx +=1
    results = []
    for _ in range(t):
        n = int(data[idx]); idx +=1
        a = list(map(int, data[idx:idx+n])); idx +=n
        prefix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + a[i]
        # Get all divisors of n
        divisors = set()
        for i in range(1, int(math.isqrt(n)) +1):
            if n % i ==0:
                divisors.add(i)
                divisors.add(n//i)
        max_diff = 0
        for k in divisors:
            if k ==0:
                continue
            num_groups = n//k
            max_sum = -1
            min_sum = 1<<60
            for g in range(num_groups):
                total = prefix[(g+1)*k] - prefix[g*k]
                if total > max_sum:
                    max_sum = total
                if total < min_sum:
                    min_sum = total
            diff = max_sum - min_sum
            if diff > max_diff:
                max_diff = diff
        results.append(str(max_diff))
    print('
'.join(results))

if __name__ == "__main__":
    main()