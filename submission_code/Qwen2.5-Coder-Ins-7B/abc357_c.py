# YOUR CODE HERE
import sys

def generate_carpet(K):
    if K == 0:
        return ["#"]
    
    smaller_carpet = generate_carpet(K - 1)
    size = 3 ** K
    carpet = []
    
    for i in range(size):
        if i // (size // 3) == 1 and i % (size // 3) == 1:
            carpet.append("." * size)
        else:
            row = []
            for j in range(size):
                if j // (size // 3) == 1 and j % (size // 3) == 1:
                    row.append(".")
                else:
                    row.append(smaller_carpet[i % (size // 3)][j % (size // 3)])
            carpet.append("".join(row))
    
    return carpet

def main():
    N = int(sys.stdin.read().strip())
    carpet = generate_carpet(N)
    for line in carpet:
        print(line)

if __name__ == "__main__":
    main()