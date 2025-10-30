import sys
import bisect

def main():
    max_M = 1000000
    n = max_M
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i : n+1 : i] = [False] * ((n - i*i) // i + 1)
    primes = [i for i in range(n + 1) if sieve[i]]
    
    valid_M = set()
    n_primes = len(primes)
    
    for i in range(n_primes):
        p = primes[i]
        temp = p
        while temp <= max_M:
            j = i + 1
            while j < n_primes:
                q = primes[j]
                if temp * q > max_M:
                    break
                current = temp * q
                while current <= max_M:
                    valid_M.add(current)
                    if current > max_M // q:
                        break
                    current *= q
                j += 1
            if temp > max_M // p:
                break
            temp *= p

    candidates = sorted([m * m for m in valid_M])
    
    data = sys.stdin.read().split()
    if not data:
        return
    Q = int(data[0])
    queries = list(map(int, data[1:1 + Q]))
    
    for A in queries:
        idx = bisect.bisect_right(candidates, A)
        if idx > 0:
            print(candidates[idx - 1])
        else:
            print(0)

if __name__ == "__main__":
    main()