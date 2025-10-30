# YOUR CODE HERE
def max_decorated_buildings(N, H):
    max_count = 1
    for i in range(N):
        for j in range(i+1, N):
            if H[i] == H[j]:
                d = j - i
                count = 2
                current = j + d
                while current < N and H[current] == H[i]:
                    count += 1
                    current += d
                if count > max_count:
                    max_count = count
    return max_count

# Read input
N = int(input())
H = list(map(int, input().split()))

# Compute the result
result = max_decorated_buildings(N, H)

# Print the result
print(result)