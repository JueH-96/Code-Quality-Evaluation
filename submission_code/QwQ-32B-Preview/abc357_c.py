def generate_carpet(N):
    if N == 0:
        return ['#']
    else:
        sub_carpet = generate_carpet(N - 1)
        size = len(sub_carpet)
        # Top row: three sub-carpets side by side
        top = [sub_carpet[i] * 3 for i in range(size)]
        # Middle row: left and right sub-carpets, central block is all '.'
        middle = [sub_carpet[i] + '.' * size + sub_carpet[i] for i in range(size)]
        # Bottom row: three sub-carpets side by side
        bottom = [sub_carpet[i] * 3 for i in range(size)]
        # Full carpet: top + middle + bottom
        full_carpet = top + middle + bottom
        return full_carpet

# Read input
N = int(input())

# Generate the carpet
carpet = generate_carpet(N)

# Print the carpet
for row in carpet:
    print(row)