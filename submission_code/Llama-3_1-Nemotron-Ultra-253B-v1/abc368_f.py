import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    max_A = 10**5
    spf = [0] * (max_A + 1)
    for i in range(2, max_A + 1):
        if spf[i] == 0:
            spf[i] = i
            for j in range(i * i, max_A + 1, i):
                if spf[j] == 0:
                    spf[j] = i
    
    xor_sum = 0
    for x in A:
        s = 0
        tmp = x
        while tmp > 1:
            p = spf[tmp]
            cnt = 0
            while tmp % p == 0:
                cnt += 1
                tmp //= p
            s += cnt
        xor_sum ^= s
    
    print("Anna" if xor_sum != 0 else "Bruno")

if __name__ == "__main__":
    main()