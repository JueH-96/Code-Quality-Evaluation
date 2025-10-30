import sys

def mobius(n):
    if n == 1:
        return 1
    facs = []
    # Check divisibility by 2
    if n % 2 == 0:
        n = n // 2
        if n % 2 == 0:
            return 0
        facs.append(2)
    # Check odd factors
    i = 3
    while i * i <= n:
        if n % i == 0:
            n = n // i
            if n % i == 0:
                return 0
            facs.append(i)
        i += 2
    if n > 1:
        facs.append(n)
    return (-1) ** len(facs)

def nth_root(n, k):
    if n == 0:
        return 0
    if n == 1:
        return 1
    low = 1
    high = 1
    # Find upper bound
    while True:
        try:
            if high ** k <= n:
                high *= 2
            else:
                break
        except OverflowError:
            break
    # Binary search
    while low < high:
        mid = (low + high + 1) // 2
        try:
            val = mid ** k
        except OverflowError:
            val = float('inf')
        if val <= n:
            low = mid
        else:
            high = mid - 1
    return low

def main():
    N = int(sys.stdin.readline().strip())
    if N < 1:
        print(0)
        return
    
    sum_terms = 0
    k_max = 0
    if N >= 2:
        while True:
            next_val = 2 ** (k_max + 1)
            if next_val > N:
                break
            k_max += 1
    
    for k in range(2, k_max + 1):
        x = nth_root(N, k)
        if x < 2:
            continue
        mu = mobius(k)
        sum_terms += (-mu) * (x - 1)
    
    total = 1 + sum_terms
    print(total)

if __name__ == "__main__":
    main()