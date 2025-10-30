def generate_carpet(n):
    if n == 0:
        return ['#']
    
    smaller_carpet = generate_carpet(n - 1)
    size = 3 ** (n - 1)
    
    top_bottom = [row * 3 for row in smaller_carpet]
    middle = [row + '.' * size + row for row in smaller_carpet]
    
    return top_bottom + middle + top_bottom

N = int(input())
carpet = generate_carpet(N)
for row in carpet:
    print(row)