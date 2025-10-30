# Read the input
A, M, L, R = map(int, input().split())

# Calculate the number of Christmas trees between Takahashi and Aoki
start = (L - A + M - 1) // M
end = (R - A) // M + 1
count = end - start

# Print the result
print(count)