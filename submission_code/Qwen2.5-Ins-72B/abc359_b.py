# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    count = 0
    for i in range(1, N + 1):
        indices = [j for j, x in enumerate(A) if x == i]
        if abs(indices[0] - indices[1]) == 2:
            count += 1
    print(count)

if __name__ == "__main__":
    main()