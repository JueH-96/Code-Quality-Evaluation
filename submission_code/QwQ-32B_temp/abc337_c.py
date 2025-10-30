import sys

def main():
    n = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    next_arr = [0] * (n + 1)
    front = 0
    for i in range(1, n + 1):
        a = A[i - 1]
        if a == -1:
            front = i
        else:
            next_arr[a] = i
    order = []
    current = front
    while current != 0:
        order.append(current)
        current = next_arr[current]
    print(' '.join(map(str, order)))

if __name__ == "__main__":
    main()