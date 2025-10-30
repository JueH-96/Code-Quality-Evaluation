# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and M
    N, M = map(int, data[0].split())
    
    # Read products
    products = []
    for i in range(1, N + 1):
        line = list(map(int, data[i].split()))
        P_i = line[0]
        C_i = line[1]
        F_i = set(line[2:2 + C_i])
        products.append((P_i, F_i))
    
    # Check for strictly superior product
    for i in range(N):
        for j in range(N):
            if i != j:
                P_i, F_i = products[i]
                P_j, F_j = products[j]
                
                # Check price condition
                if P_i >= P_j:
                    # Check function inclusion
                    if F_i.issubset(F_j):
                        # Check strict superiority
                        if P_i > P_j or len(F_j - F_i) > 0:
                            print("Yes")
                            return
    
    print("No")

if __name__ == "__main__":
    main()