# Read the number of sellers and buyers
N, M = map(int, input().split())

# Read the minimum prices for sellers
A = list(map(int, input().split()))

# Read the maximum prices for buyers
B = list(map(int, input().split()))

# Initialize the answer to the maximum possible value
X = max(max(A), max(B)) + 1

# Sort the lists to make the search faster
A.sort()
B.sort()

# Initialize pointers for sellers and buyers
seller_index = 0
buyer_index = 0

# While there are sellers and buyers to consider
while seller_index < N and buyer_index < M:
    # If the current seller's price is less than or equal to the current buyer's price
    if A[seller_index] <= B[buyer_index]:
        # Update the answer to be the maximum of the current seller's price and the current answer
        X = min(X, A[seller_index])
        # Move to the next seller
        seller_index += 1
    else:
        # Move to the next buyer
        buyer_index += 1

# Print the answer
print(X)