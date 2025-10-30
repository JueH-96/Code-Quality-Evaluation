# YOUR CODE HERE
import sys
input = sys.stdin.read

MOD = 998244353

def main():
    data = input().split()
    N = int(data[0])
    A = data[1:]
    
    result = 0
    
    for i in range(N - 1):
        for j in range(i + 1, N):
            f_ij = int(A[i] + A[j])
            result = (result + f_ij) % MOD
    
    print(result)

if __name__ == "__main__":
    main()