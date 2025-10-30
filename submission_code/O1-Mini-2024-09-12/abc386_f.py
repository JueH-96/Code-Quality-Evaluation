import sys

def can_transform(K, S, T):
    m, n = len(S), len(T)
    # If the difference in lengths is more than K, it's impossible
    if abs(m - n) > K:
        return False

    # Ensure S is the shorter string
    if m > n:
        S, T = T, S
        m, n = n, m

    # Initialize the dictionary to hold the farthest j for each diagonal k
    # k = i - j, shifted by K to handle negative indices
    V = [ -1 ] * (2*K + 1)
    V[K + 1] = 0  # Initial diagonal

    for d in range(0, K + 1):
        for k in range(-d, d + 1, 2):
            idx = k + K
            if k == -d or (k != d and V[idx -1] < V[idx +1]):
                x = V[idx +1]
            else:
                x = V[idx -1] + 1
            y = x - k
            # Extend the match as far as possible
            while x < m and y < n and S[x] == T[y]:
                x += 1
                y += 1
            V[idx] = x
            # Check if we've reached the end
            if x >= m and y >= n:
                return True
    return False

def main():
    input = sys.stdin.read().split()
    K = int(input[0])
    S = input[1]
    T = input[2]
    if can_transform(K, S, T):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()