# YOUR CODE HERE
import sys

def calculate_v_n_mod(n, mod=998244353):
    # Convert N to string
    n_str = str(n)
    # Length of the string representation of N
    len_n_str = len(n_str)

    # V_N is formed by concatenating N exactly N times
    # Instead of forming the full number, we use modular arithmetic to keep the number manageable
    v_n_mod = 0
    for _ in range(n):
        v_n_mod = (v_n_mod * (10 ** len_n_str) + int(n_str)) % mod

    return v_n_mod

def main():
    input = sys.stdin.read()
    n = int(input.strip())
    result = calculate_v_n_mod(n)
    print(result)

if __name__ == "__main__":
    main()