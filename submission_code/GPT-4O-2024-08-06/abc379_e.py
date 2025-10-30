# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    
    total_sum = 0
    
    # Precompute powers of 10 to avoid recomputation
    powers_of_10 = [1] * N
    for i in range(1, N):
        powers_of_10[i] = powers_of_10[i - 1] * 10
    
    # Calculate the sum of all f(i, j)
    for i in range(N):
        current_value = 0
        for j in range(i, N):
            current_value = current_value * 10 + int(S[j])
            total_sum += current_value
    
    print(total_sum)

if __name__ == "__main__":
    main()