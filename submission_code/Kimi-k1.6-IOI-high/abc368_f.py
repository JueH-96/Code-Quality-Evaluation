import sys

def main():
    max_x = 10**5
    # Precompute proper divisors for each number up to 10^5
    divisors = [[] for _ in range(max_x + 1)]
    for d in range(1, max_x + 1):
        for m in range(2 * d, max_x + 1, d):
            divisors[m].append(d)
    
    # Precompute Grundy numbers
    grundy = [0] * (max_x + 1)
    grundy[1] = 0
    for x in range(2, max_x + 1):
        seen = set()
        for d in divisors[x]:
            seen.add(grundy[d])
        mex = 0
        while mex in seen:
            mex += 1
        grundy[x] = mex
    
    # Read input
    input = sys.stdin.read().split()
    n = int(input[0])
    a_list = list(map(int, input[1:n+1]))
    
    # Calculate XOR sum of Grundy numbers
    xor_sum = 0
    for a in a_list:
        xor_sum ^= grundy[a]
    
    print("Anna" if xor_sum != 0 else "Bruno")

if __name__ == "__main__":
    main()