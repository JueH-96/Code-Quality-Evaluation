# YOUR CODE HERE
MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    if M == 1:
        if N == 1:
            print(1)
        else:
            print(0)
        return
    
    if N == 1:
        print(M % MOD)
        return
    
    # Calculate (M-1)^N
    power = pow(M-1, N, MOD)
    
    # Calculate (-1)^N
    sign = (-1) ** N
    
    # Calculate (M-1) * (-1)^N
    term = (M-1) * sign % MOD
    
    # Total ways = (M-1)^N + (M-1) * (-1)^N
    total = (power + term) % MOD
    
    print(total)

if __name__ == "__main__":
    main()