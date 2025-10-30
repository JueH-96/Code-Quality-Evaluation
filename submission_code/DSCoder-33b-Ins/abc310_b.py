# YOUR CODE HERE

import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    products = []
    for _ in range(N):
        P, C, *F = map(int, sys.stdin.readline().split())
        products.append((P, set(F)))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if products[i][0] >= products[j][0] and products[i][1].issubset(products[j][1]):
                if products[i][0] > products[j][0] or not products[j][1].issubset(products[i][1]):
                    print('Yes')
                    return
    print('No')

if __name__ == '__main__':
    main()