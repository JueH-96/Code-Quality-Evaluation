# YOUR CODE HERE
def solve(N):
    mod = 998244353
    num_str = str(N)
    full_str = num_str * N
    V_N = int(full_str) % mod
    return V_N

if __name__ == "__main__":
    N = int(input().strip())
    print(solve(N))