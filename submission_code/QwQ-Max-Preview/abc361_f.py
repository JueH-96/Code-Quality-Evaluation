import sys

def max_a_for_exponent(n, e):
    if e == 0:
        return 0
    low = 2
    high = n
    best = 1
    while low <= high:
        mid = (low + high) // 2
        product = 1
        overflow = False
        for _ in range(e):
            product *= mid
            if product > n:
                overflow = True
                break
        if overflow:
            high = mid - 1
        else:
            best = mid
            low = mid + 1
    return best

def count_perfect_powers_up_to(m):
    if m < 2:
        return 0
    s = set()
    max_b = 1
    while (2 ** (max_b + 1)) <= m:
        max_b += 1
    for b in range(2, max_b + 1):
        low = 2
        high = m
        a_max_b = 1
        while low <= high:
            mid = (low + high) // 2
            product = 1
            overflow = False
            for _ in range(b):
                product *= mid
                if product > m:
                    overflow = True
                    break
            if overflow:
                high = mid - 1
            else:
                a_max_b = mid
                low = mid + 1
        if a_max_b >= 2:
            for a in range(2, a_max_b + 1):
                s.add(pow(a, b))
    return len(s)

def main():
    N = int(sys.stdin.readline().strip())
    if N < 1:
        print(0)
        return
    answer = 1  # 1 is counted once
    max_e = 60
    for e in range(2, max_e + 1):
        a_max = max_a_for_exponent(N, e)
        if a_max < 2:
            break
        m = a_max
        count = count_perfect_powers_up_to(m)
        answer += (m - 1 - count)
    print(answer)

if __name__ == "__main__":
    main()