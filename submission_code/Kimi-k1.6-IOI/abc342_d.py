from collections import defaultdict

def main():
    max_num = 200000
    spf = list(range(max_num + 1))
    for i in range(2, int(max_num**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, max_num + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    n = int(input())
    a = list(map(int, input().split()))
    z = a.count(0)
    non_zero = [x for x in a if x != 0]
    
    freq = defaultdict(int)
    for x in non_zero:
        sf = 1
        current = x
        while current != 1:
            p = spf[current]
            cnt = 0
            while current % p == 0:
                cnt += 1
                current //= p
            if cnt % 2 != 0:
                sf *= p
        freq[sf] += 1
    
    sum_non_zero = 0
    for cnt in freq.values():
        sum_non_zero += cnt * (cnt - 1) // 2
    
    zero_pairs = z * (z - 1) // 2 + z * (n - z)
    total = zero_pairs + sum_non_zero
    print(total)

if __name__ == "__main__":
    main()