# YOUR CODE HERE
def solve():
    n = int(input())

    def generate_carpet(k):
        if k == 0:
            return [['#']]
        
        size = 3**k
        carpet = [['.' for _ in range(size)] for _ in range(size)]
        
        smaller_carpet = generate_carpet(k-1)
        smaller_size = 3**(k-1)
        
        for i in range(smaller_size):
            for j in range(smaller_size):
                carpet[i][j] = smaller_carpet[i][j]
                carpet[i][j + smaller_size * 2] = smaller_carpet[i][j]
                carpet[i + smaller_size * 2][j] = smaller_carpet[i][j]
                carpet[i + smaller_size * 2][j + smaller_size * 2] = smaller_carpet[i][j]
                carpet[i + smaller_size][j] = smaller_carpet[i][j]
                carpet[i + smaller_size][j + smaller_size * 2] = smaller_carpet[i][j]
                carpet[i][j + smaller_size] = smaller_carpet[i][j]
                carpet[i + smaller_size * 2][j + smaller_size] = smaller_carpet[i][j]
        
        return carpet

    carpet = generate_carpet(n)
    for row in carpet:
        print("".join(row))

solve()