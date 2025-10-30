import sys

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    # Precompute 10^length mod MOD for each element in A
    pow_10_list = []
    for a in A:
        length = len(str(a))
        pow_10 = pow(10, length, MOD)
        pow_10_list.append(pow_10)
    
    sum_part1 = 0
    sum_part2 = 0
    prefix_sum = 0
    count = 0
    
    for a_j, pow_10_j in zip(A, pow_10_list):
        # Update sum_part1: prefix_sum * pow_10_j
        term1 = (prefix_sum * pow_10_j) % MOD
        sum_part1 = (sum_part1 + term1) % MOD
        
        # Update sum_part2: a_j * count
        term2 = (a_j * count) % MOD
        sum_part2 = (sum_part2 + term2) % MOD
        
        # Update prefix_sum and count
        prefix_sum = (prefix_sum + a_j) % MOD
        count += 1
    
    total = (sum_part1 + sum_part2) % MOD
    print(total)

if __name__ == "__main__":
    main()