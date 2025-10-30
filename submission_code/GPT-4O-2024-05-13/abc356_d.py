# YOUR CODE HERE
def popcount(x):
    return bin(x).count('1')

def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    N = int(data[0])
    M = int(data[1])
    
    MOD = 998244353
    total_sum = 0
    
    for k in range(N + 1):
        total_sum += popcount(k & M)
        total_sum %= MOD
    
    print(total_sum)

if __name__ == "__main__":
    main()