# YOUR CODE HERE
import sys

def main():
    import sys

    # Read all input
    data = sys.stdin.read().split()
    N = int(data[0])
    K = list(map(int, data[1:N+1]))
    total = sum(K)
    min_max = total  # Initialize with the total as the worst case

    for mask in range(1 << N):
        sumA = 0
        for i in range(N):
            if mask & (1 << i):
                sumA += K[i]
        current_max = max(sumA, total - sumA)
        if current_max < min_max:
            min_max = current_max

    print(min_max)

if __name__ == "__main__":
    main()