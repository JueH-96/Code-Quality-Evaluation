import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    max_limit = 10**5
    
    # Precompute divisors for each number up to max_limit
    divisors = [[] for _ in range(max_limit + 1)]
    for d in range(1, max_limit + 1):
        for multiple in range(d, max_limit + 1, d):
            divisors[multiple].append(d)
    
    # Precompute Grundy numbers
    g = [0] * (max_limit + 1)
    for n in range(2, max_limit + 1):
        proper_divs = divisors[n][:-1]  # exclude the number itself
        s = set()
        for d in proper_divs:
            s.add(g[d])
        mex = 0
        while mex in s:
            mex += 1
        g[n] = mex
    
    # Calculate XOR sum of Grundy numbers for the input array
    xor_sum = 0
    for a in A:
        xor_sum ^= g[a]
    
    print("Anna" if xor_sum != 0 else "Bruno")

if __name__ == "__main__":
    main()