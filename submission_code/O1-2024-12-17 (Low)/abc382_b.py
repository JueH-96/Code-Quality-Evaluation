def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, D = map(int, data[:2])
    S = data[2]

    boxes = list(S)
    remaining_eats = D

    # Eat a cookie from the rightmost box day by day
    for i in reversed(range(N)):
        if boxes[i] == '@':
            boxes[i] = '.'
            remaining_eats -= 1
            if remaining_eats == 0:
                break

    print("".join(boxes))

if __name__ == "__main__":
    main()