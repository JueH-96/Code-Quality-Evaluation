# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    M = int(input[index + 1])
    index += 2
    products = []

    for _ in range(N):
        P = int(input[index])
        C = int(input[index + 1])
        F = list(map(int, input[index + 2:index + 2 + C]))
        products.append((P, set(F)))
        index += 2 + C

    for i in range(N):
        for j in range(N):
            if i != j:
                P_i, F_i = products[i]
                P_j, F_j = products[j]
                if P_i >= P_j and F_i.issubset(F_j) and (P_i > P_j or F_j - F_i):
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()