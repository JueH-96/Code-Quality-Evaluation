def get_divisors(n):
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    divisors.sort()
    return divisors

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx])
    idx += 1
    for _ in range(t):
        n = int(input[idx])
        idx += 1
        a = list(map(int, input[idx:idx+n]))
        idx += n
        divisors = get_divisors(n)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + a[i]
        max_diff = 0
        for k in divisors:
            m = n // k
            truck_sums = [prefix[p*k + k] - prefix[p*k] for p in range(m)]
            if truck_sums:
                max_sum = max(truck_sums)
                min_sum = min(truck_sums)
                diff = max_sum - min_sum
                if diff > max_diff:
                    max_diff = diff
        print(max_diff)

if __name__ == "__main__":
    main()