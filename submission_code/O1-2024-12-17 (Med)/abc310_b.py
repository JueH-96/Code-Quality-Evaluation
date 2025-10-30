def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    idx = 2
    products = []
    
    # Read product information
    for _ in range(N):
        P_i = int(data[idx]); idx += 1
        C_i = int(data[idx]); idx += 1
        features = set(map(int, data[idx:idx + C_i]))
        idx += C_i
        products.append((P_i, features))
    
    # Check each pair (i, j)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            
            price_i, features_i = products[i]
            price_j, features_j = products[j]
            
            # Conditions:
            # 1) P_i >= P_j
            # 2) features_i is subset of features_j (features_j has all of i's features)
            # 3) either P_i > P_j or j-th product has extra features (not in i-th product)
            if (price_i >= price_j and
                features_i.issubset(features_j) and
                (price_i > price_j or len(features_j - features_i) > 0)):
                print("Yes")
                return
    
    print("No")

def main_wrapper():
    main()

# Call main() at the end
if __name__ == "__main__":
    main_wrapper()