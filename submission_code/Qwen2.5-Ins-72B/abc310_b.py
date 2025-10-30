# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    index = 2
    products = []
    
    for _ in range(N):
        P = int(data[index])
        C = int(data[index + 1])
        F = set(int(data[index + 2 + i]) for i in range(C))
        products.append((P, F))
        index += 2 + C
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            Pi, Fi = products[i]
            Pj, Fj = products[j]
            if Pi >= Pj and Fi.issubset(Fj) and (Pi > Pj or Fj - Fi):
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()