import sys
from itertools import permutations

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1

    functions = []
    for _ in range(N):
        A = int(input[idx])
        idx += 1
        B = int(input[idx])
        idx += 1
        functions.append((A, B))

    if K == 0:
        print(1)
        return

    max_val = 0

    for perm in permutations(functions, K):
        # Calculate product_A: product of all A's in the permutation
        product_A = 1
        for (A, B) in perm:
            product_A *= A
        
        # Prepare application order (reversed permutation)
        application_order = list(reversed(perm))
        K_len = len(application_order)
        
        # Precompute product_after for each position
        product_after = [1] * K_len
        for i in range(K_len - 2, -1, -1):
            product_after[i] = product_after[i + 1] * application_order[i + 1][0]
        
        # Calculate sum_total
        sum_total = 0
        for i in range(K_len):
            sum_total += application_order[i][1] * product_after[i]
        
        # Calculate total
        total = product_A * 1 + sum_total  # initial value is 1
        
        if total > max_val:
            max_val = total

    print(max_val)

if __name__ == '__main__':
    main()