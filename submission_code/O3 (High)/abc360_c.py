import sys

def main() -> None:
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))
    W = list(map(int, input().split()))

    # max_in_box[b] : heaviest item currently in box b (1-indexed).
    max_in_box = [0] * (N + 1)

    total_weight = 0
    for box, weight in zip(A, W):
        total_weight += weight
        if weight > max_in_box[box]:
            max_in_box[box] = weight

    kept_weight = sum(max_in_box)       # weight of all items we leave where they are
    min_cost    = total_weight - kept_weight

    print(min_cost)


if __name__ == "__main__":
    main()