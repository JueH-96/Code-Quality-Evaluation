def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    S = []
    T = []
    
    index = N + 1
    for i in range(N-1):
        S.append(int(data[index]))
        T.append(int(data[index+1]))
        index += 2
    
    # We will process from the first country to the last one
    for i in range(N-1):
        # Calculate the maximum number of transactions we can perform with country i
        max_transactions = A[i] // S[i]
        # Calculate the total currency we can transfer to country i+1
        currency_to_next = max_transactions * T[i]
        # Add this currency to country i+1
        A[i+1] += currency_to_next
        # Subtract the spent currency from country i
        A[i] -= max_transactions * S[i]
    
    # The result is the amount of currency in the last country
    print(A[-1])