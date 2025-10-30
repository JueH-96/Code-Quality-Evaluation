import sys
import bisect

def main():
    max_k = 10**6
    is_prime = [True] * (max_k + 1)
    primes = []
    is_prime[0] = is_prime[1] = False
    for i in range(2, max_k + 1):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if p * i > max_k:
                break
            is_prime[p * i] = False
            if i % p == 0:
                break
    
    omega = [0] * (max_k + 1)
    for p in primes:
        for j in range(p, max_k + 1, p):
            omega[j] += 1
    
    L = []
    for k in range(1, max_k + 1):
        if omega[k] == 2:
            num = k * k
            L.append(num)
    
    data = sys.stdin.read().split()
    if not data:
        return
    q = int(data[0])
    queries = list(map(int, data[1:1 + q]))
    
    res = []
    for A in queries:
        idx = bisect.bisect_right(L, A) - 1
        res.append(str(L[idx]))
    
    sys.stdout.write("
".join(res))

if __name__ == "__main__":
    main()