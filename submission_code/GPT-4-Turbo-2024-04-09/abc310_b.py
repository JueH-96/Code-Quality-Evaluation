def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    M = int(data[index + 1])
    index += 2
    
    products = []
    
    for _ in range(N):
        P = int(data[index])
        C = int(data[index + 1])
        functions = set(map(int, data[index + 2: index + 2 + C]))
        index += 2 + C
        products.append((P, functions))
    
    for i in range(N):
        Pi, Fi = products[i]
        for j in range(N):
            if i != j:
                Pj, Fj = products[j]
                if Pi >= Pj and Fi.issubset(Fj) and (Pi > Pj or Fj.difference(Fi)):
                    print("Yes")
                    return
    
    print("No")

if __name__ == "__main__":
    main()