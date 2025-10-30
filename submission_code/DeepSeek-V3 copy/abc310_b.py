# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    products = []
    index = 2
    for _ in range(N):
        P = int(data[index])
        C = int(data[index+1])
        F = set(map(int, data[index+2:index+2+C]))
        products.append((P, F))
        index += 2 + C
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            P_i, F_i = products[i]
            P_j, F_j = products[j]
            if P_i >= P_j and F_i.issubset(F_j):
                if P_i > P_j or len(F_j - F_i) > 0:
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()