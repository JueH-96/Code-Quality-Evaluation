import sys

def read_input():
    N, M = map(int, sys.stdin.readline().split())
    products = []
    for _ in range(N):
        P, C = map(int, sys.stdin.readline().split())
        F = list(map(int, sys.stdin.readline().split()))
        products.append((P, C, set(F)))
    return products

def solve(products):
    products.sort(reverse=True)
    max_functions = set()
    for P, C, F in products:
        if F >= max_functions:
            max_functions = F
        else:
            return "Yes"
    return "No"

def main():
    products = read_input()
    print(solve(products))

if __name__ == "__main__":
    main()