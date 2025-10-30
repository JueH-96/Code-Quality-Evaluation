import sys

def solve_test_case():
    n = int(input())
    a = list(map(int, input().split()))
    
    max_product = 0
    for i in range(n):
        temp = a[:]
        if temp[i] < 9:
            temp[i] += 1
            product = 1
            for x in temp:
                product *= x
            max_product = max(max_product, product)
    
    return max_product

t = int(input())
for _ in range(t):
    print(solve_test_case())