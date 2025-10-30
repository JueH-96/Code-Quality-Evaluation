# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    S = []
    T = []
    
    index = N + 1
    for i in range(N - 1):
        S.append(int(data[index]))
        T.append(int(data[index + 1]))
        index += 2
    
    # Process each country from 1 to N-1
    for i in range(N - 1):
        # Calculate how many times we can perform the operation
        max_operations = A[i] // S[i]
        # Update the currency in the next country
        A[i + 1] += max_operations * T[i]
        # Reduce the currency in the current country
        A[i] -= max_operations * S[i]
    
    # The result is the amount of currency in the last country
    print(A[N - 1])

main()