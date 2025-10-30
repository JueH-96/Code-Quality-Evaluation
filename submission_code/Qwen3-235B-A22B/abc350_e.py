import sys
sys.setrecursionlimit(1 << 25)

def main():
    N, A, X, Y = map(int, sys.stdin.read().split())
    memo = {}

    def compute(n):
        if n == 0:
            return 0.0
        if n in memo:
            return memo[n]
        
        # Option 1: Pay X and replace with n // A
        next_n_a = n // A
        cost_a = X + compute(next_n_a)
        
        # Option 2: Compute die effective cost
        sum_b2_6 = 0.0
        for b in range(2, 7):
            next_n_b = n // b
            sum_b2_6 += compute(next_n_b)
        cost_die = (6 * Y + sum_b2_6) / 5.0
        
        # Choose minimal cost
        result = min(cost_a, cost_die)
        memo[n] = result
        return result

    result = compute(N)
    print("{0:.15f}".format(result))

if __name__ == "__main__":
    main()