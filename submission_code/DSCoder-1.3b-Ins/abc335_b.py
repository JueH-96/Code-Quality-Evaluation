def print_triplets(n):
    for i in range(n+1):
        for j in range(n+1):
            for k in range(n+1):
                if i + j + k <= n:
                    print(f"{i} {j} {k}")

# Read the input
n = int(input())

# Call the function
print_triplets(n)