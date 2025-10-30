def solve():
    n, m = map(int, input().split())
    products = []
    for _ in range(n):
        line = list(map(int, input().split()))
        price = line[0]
        num_functions = line[1]
        functions = set(line[2:])
        products.append({'price': price, 'functions': functions})
    
    found_superior = False
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            product_i = products[i]
            product_j = products[j]
            price_i = product_i['price']
            price_j = product_j['price']
            functions_i = product_i['functions']
            functions_j = product_j['functions']
            
            condition1 = price_i >= price_j
            condition2 = functions_i.issubset(functions_j)
            condition3 = (price_i > price_j) or (functions_i < functions_j)
            
            if condition1 and condition2 and condition3:
                found_superior = True
                break
        if found_superior:
            break
            
    if found_superior:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()