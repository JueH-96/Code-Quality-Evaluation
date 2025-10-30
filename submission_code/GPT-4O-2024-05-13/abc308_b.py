# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    C = data[2:2+N]
    D = data[2+N:2+N+M]
    P = list(map(int, data[2+N+M:2+N+M+M+1]))
    
    P_0 = P[0]
    P_dict = {D[i]: P[i+1] for i in range(M)}
    
    total_price = 0
    for color in C:
        if color in P_dict:
            total_price += P_dict[color]
        else:
            total_price += P_0
    
    print(total_price)

if __name__ == "__main__":
    main()