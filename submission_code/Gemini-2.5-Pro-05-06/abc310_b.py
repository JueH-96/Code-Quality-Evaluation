import sys

def main():
    N, M_unused = map(int, sys.stdin.readline().split())
    
    products = []
    for _ in range(N):
        line_parts = list(map(int, sys.stdin.readline().split()))
        price = line_parts[0]
        # C_i = line_parts[1] # Number of functions for this product
        # The actual functions start from index 2 of line_parts
        features = set(line_parts[2:])
        products.append({'price': price, 'features': features})
        
    for i in range(N):
        P_i = products[i]['price']
        S_i = products[i]['features']
        
        for j in range(N):
            # As analyzed in thought process, an explicit `if i == j: continue` is not needed.
            # If i == j, condition 3 will evaluate to False, so product i won't be
            # considered strictly superior to itself.
            
            P_j = products[j]['price']
            S_j = products[j]['features']

            # Condition 1: P_i >= P_j
            if not (P_i >= P_j):
                continue

            # Condition 2: The j-th product has all functions of the i-th product.
            # This means S_i is a subset of S_j.
            if not S_i.issubset(S_j):
                continue
            
            # At this point, conditions 1 (P_i >= P_j) and 2 (S_i is subset of S_j) are met.
            # Now, check condition 3:
            # (P_i > P_j) OR (the j-th product has one or more functions that the i-th product lacks)
            
            # "the j-th product has one or more functions that the i-th product lacks"
            # means S_j \ S_i is not an empty set.
            # Given that S_i.issubset(S_j) (Condition 2) is true,
            # S_j \ S_i is not empty if and only if S_i is a proper subset of S_j.
            # S_i is a proper subset of S_j if and only if len(S_i) < len(S_j)
            # (this relies on S_i.issubset(S_j) already being true).
            
            # So, the third overall condition becomes: (P_i > P_j) OR (len(S_i) < len(S_j))
            if (P_i > P_j) or (len(S_i) < len(S_j)):
                # All three conditions are satisfied. Product i is strictly superior to product j.
                sys.stdout.write("Yes
")
                return
                
    # If loops complete, no such pair (i, j) was found.
    sys.stdout.write("No
")

if __name__ == '__main__':
    main()