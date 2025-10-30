# Read the number of integers
N = int(input())

# Read the integers into a list
A = list(map(int, input().split()))

# Calculate B_i for each i and print them separated by spaces
B = [A[i] * A[i+1] for i in range(N-1)]
print(' '.join(map(str, B)))