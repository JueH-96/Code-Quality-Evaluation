import sys

def main():
    max_val = 100000
    min_prime = [0] * (max_val + 1)
    omega = [0] * (max_val + 1)
    primes = []
    
    for i in range(2, max_val + 1):
        if min_prime[i] == 0:
            min_prime[i] = i
            omega[i] = 1
            primes.append(i)
        for p in primes:
            if p > min_prime[i] or p * i > max_val:
                break
            num = p * i
            min_prime[num] = p
            omega[num] = omega[i] + 1
    
    omega[1] = 0

    data = sys.stdin.read().split()
    n = int(data[0])
    arr = list(map(int, data[1:1 + n]))
    
    xor_val = 0
    for a in arr:
        xor_val ^= omega[a]
    
    print("Anna" if xor_val != 0 else "Bruno")

if __name__ == "__main__":
    main()