def min_operations_to_divisible(n, k, a):
    # Calculate the product modulo k
    product_mod_k = 1
    for i in range(n):
        product_mod_k = (product_mod_k * (a[i] % k)) % k
    
    # If product is already divisible by k
    if product_mod_k == 0:
        return 0
    
    # Special case for k = 2
    if k == 2:
        return 1  # If product is odd, we need 1 operation to make one element even
    
    # For k = 3, 4, 5
    min_ops = float('inf')
    for i in range(n):
        for ops in range(1, k):
            new_a_i = a[i] + ops
            
            # Calculate the new product modulo k
            new_product_mod_k = 1
            for j in range(n):
                if j == i:
                    new_product_mod_k = (new_product_mod_k * (new_a_i % k)) % k
                else:
                    new_product_mod_k = (new_product_mod_k * (a[j] % k)) % k
            
            if new_product_mod_k == 0:
                min_ops = min(min_ops, ops)
    
    return min_ops

# Main function to handle input and output
def main():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        a = list(map(int, input().split()))
        print(min_operations_to_divisible(n, k, a))

if __name__ == "__main__":
    main()