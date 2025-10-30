# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:]))

    total_cost = sum(A)
    if total_cost <= M:
        print("infinite")
        return

    def can_afford(x):
        return sum(min(x, a) for a in A) <= M

    low, high = 0, max(A)
    while low < high:
        mid = (low + high + 1) // 2
        if can_afford(mid):
            low = mid
        else:
            high = mid - 1

    print(low)

if __name__ == "__main__":
    main()