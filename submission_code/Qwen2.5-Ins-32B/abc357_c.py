def generate_carpet(N):
    if N == 0:
        return ["#"]
    
    prev_carpet = generate_carpet(N - 1)
    size = 3 ** (N - 1)
    new_carpet = []
    
    for i in range(3 * size):
        if size <= i < 2 * size:
            new_row = prev_carpet[i % size] + "." * size + prev_carpet[i % size]
        else:
            new_row = prev_carpet[i % size] * 3
        new_carpet.append(new_row)
    
    return new_carpet

N = int(input())
carpet = generate_carpet(N)
for row in carpet:
    print(row)