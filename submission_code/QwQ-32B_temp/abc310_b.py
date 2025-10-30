import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    products = []
    for _ in range(n):
        parts = list(map(int, sys.stdin.readline().split()))
        p = parts[0]
        c = parts[1]
        funcs = set(parts[2:2 + c])
        products.append((p, funcs))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pi, fi = products[i]
            pj, fj = products[j]
            if pi >= pj and fi.issubset(fj):
                if (pi > pj) or (fi < fj):
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()