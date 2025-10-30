import sys

def main():
    n, m = map(int, input().split())
    products = []
    for _ in range(n):
        data = list(map(int, input().split()))
        p = data[0]
        c = data[1]
        funcs = set(data[2:])
        products.append((p, funcs))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_i = products[i][0]
            p_j = products[j][0]
            if p_i >= p_j:
                funcs_i = products[i][1]
                funcs_j = products[j][1]
                if funcs_i.issubset(funcs_j):
                    if p_i > p_j or len(funcs_j) > len(funcs_i):
                        print("Yes")
                        return
    print("No")

if __name__ == "__main__":
    main()