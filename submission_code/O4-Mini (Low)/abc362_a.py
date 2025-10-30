def main():
    import sys
    data = sys.stdin.read().split()
    R, G, B = map(int, data[:3])
    C = data[3]
    
    costs = []
    if C != "Red":
        costs.append(R)
    if C != "Green":
        costs.append(G)
    if C != "Blue":
        costs.append(B)
    
    print(min(costs))

if __name__ == "__main__":
    main()