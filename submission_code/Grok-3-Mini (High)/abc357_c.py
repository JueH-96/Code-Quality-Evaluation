def generate_carpet(K):
    if K == 0:
        return ["#"]
    else:
        small_carpet = generate_carpet(K - 1)
        size_small = len(small_carpet)
        carpet = []
        for br in range(3):  # Iterate over big rows
            for lr in range(size_small):  # Iterate over local rows within each big row
                if br == 1:
                    part_bc1_str = "." * size_small  # Middle part is white if in the middle big row
                else:
                    part_bc1_str = small_carpet[lr]  # Middle part is from small carpet otherwise
                # Parts for left and right are always from small carpet
                row_str = small_carpet[lr] + part_bc1_str + small_carpet[lr]
                carpet.append(row_str)
        return carpet

# Read input from stdin
N = int(input())

# Generate and print the carpet
carpet = generate_carpet(N)
for row in carpet:
    print(row)