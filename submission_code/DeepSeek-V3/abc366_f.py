import sys
import itertools

def main():
    N, K = map(int, sys.stdin.readline().split())
    functions = []
    for _ in range(N):
        A, B = map(int, sys.stdin.readline().split())
        functions.append((A, B))
    
    if K == 1:
        max_val = 0
        for A, B in functions:
            val = A * 1 + B
            if val > max_val:
                max_val = val
        print(max_val)
        return
    
    # Since K is small (up to 10), we can try all possible combinations
    # but with N up to 2e5, we need to optimize
    # So, we need to find the top K functions that maximize the final value
    
    # To maximize f_{p1}(f_{p2}(...f_{pK}(1)...)), we need to choose functions in a specific order
    # The order matters because the composition is not commutative
    
    # Let's consider the composition of two functions:
    # f1(f2(x)) = A1*(A2*x + B2) + B1 = A1*A2*x + A1*B2 + B1
    # So, the coefficient is A1*A2, and the constant term is A1*B2 + B1
    
    # For K functions, the final coefficient is A_{p1} * A_{p2} * ... * A_{pK}
    # and the constant term is A_{p1} * A_{p2} * ... * A_{pK-1} * B_{pK} + A_{p1} * A_{p2} * ... * A_{pK-2} * B_{pK-1} + ... + A_{p1} * B_{p2} + B_{p1}
    
    # To maximize the final value, we need to maximize the coefficient and the constant term
    
    # Since A_i >= 1, the coefficient is maximized by choosing the K functions with the largest A_i
    
    # However, the order of the functions also affects the constant term
    
    # So, we first select the K functions with the largest A_i, and then try all permutations of these K functions to find the maximum value
    
    # Step 1: Select the top K functions with the largest A_i
    # If there are ties, we can choose any of them, but to maximize the constant term, we should prefer those with larger B_i
    
    # So, we sort the functions by A_i in descending order, and then by B_i in descending order
    sorted_functions = sorted(functions, key=lambda x: (x[0], x[1]), reverse=True)
    top_K = sorted_functions[:K]
    
    # Step 2: Try all permutations of these K functions
    max_total = 0
    for perm in itertools.permutations(top_K):
        current = 1
        for A, B in reversed(perm):
            current = A * current + B
        if current > max_total:
            max_total = current
    
    print(max_total)

if __name__ == "__main__":
    main()