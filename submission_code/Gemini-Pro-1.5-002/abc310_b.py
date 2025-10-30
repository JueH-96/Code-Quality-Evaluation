# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    products = []
    for _ in range(n):
        line = list(map(int, input().split()))
        products.append((line[0], set(line[2:])))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            pi, fi = products[i]
            pj, fj = products[j]

            if pi >= pj and fj.issuperset(fi):
                if pi > pj or not fi.issuperset(fj):
                    print("Yes")
                    return

    print("No")

solve()