import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    if not A:
        print("Bruno")
        return
    
    max_num = 10**5
    spf = [0] * (max_num + 1)
    
    for i in range(2, max_num + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= max_num:
                for j in range(i * i, max_num + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
    # Handle remaining primes
    for i in range(2, max_num + 1):
        if spf[i] == 0:
            spf[i] = i
    
    xor = 0
    for x in A:
        tmp = x
        sum_exp = 0
        while tmp != 1:
            p = spf[tmp]
            count = 0
            while tmp % p == 0:
                tmp = tmp // p
                count += 1
            sum_exp += count
        xor ^= sum_exp
    
    if xor != 0:
        print("Anna")
    else:
        print("Bruno")

if __name__ == "__main__":
    main()