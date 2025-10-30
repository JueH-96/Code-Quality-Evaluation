# YOUR CODE HERE
def max_strawberries(N, K, S):
    i = 0
    strawberries = 0
    while i <= N - K:
        # Check if we have K consecutive 'O's starting from index i
        if S[i:i+K] == 'O' * K:
            strawberries += 1
            # Move i forward by K to avoid overlapping
            i += K
        else:
            # Move to the next tooth
            i += 1
    return strawberries

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    result = max_strawberries(N, K, S)
    print(result)