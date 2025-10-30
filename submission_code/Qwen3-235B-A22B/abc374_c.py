import sys

def main():
    n = int(sys.stdin.readline())
    k = list(map(int, sys.stdin.readline().split()))
    total = sum(k)
    subset_sums = [0]
    for num in k:
        new_sums = [s + num for s in subset_sums]
        subset_sums += new_sums
    min_max = total
    for s in subset_sums:
        current = max(s, total - s)
        if current < min_max:
            min_max = current
    print(min_max)

if __name__ == "__main__":
    main()