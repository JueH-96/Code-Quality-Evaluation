def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    idx = 2
    
    products = []
    for _ in range(N):
        P = int(data[idx]); idx += 1
        C = int(data[idx]); idx += 1
        features = set(map(int, data[idx:idx + C]))
        idx += C
        products.append((P, features))
    
    # Check if there exists a pair (i, j) such that product i is strictly superior to product j
    for i in range(N):
        Pi, Fi = products[i]
        for j in range(N):
            if i == j:
                continue
            Pj, Fj = products[j]
            # Conditions:
            # 1) Pi >= Pj
            # 2) Fi is subset of Fj  (the jth product has all functions of the ith product)
            # 3) Pi > Pj or Fj has more functions than Fi
            #    (i.e. Fi != Fj since Fi ⊆ Fj)
            if Pi >= Pj and Fi.issubset(Fj) and (Pi > Pj or Fi != Fj):
                print("Yes")
                return
    print("No")

# Do not forget to call main()
main()