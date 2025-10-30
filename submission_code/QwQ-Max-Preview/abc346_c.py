def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    k = int(input[1])
    a = list(map(int, input[2:2+n]))
    unique_a = set(a)
    sum_present = sum(x for x in unique_a if x <= k)
    total = k * (k + 1) // 2
    print(total - sum_present)

if __name__ == "__main__":
    main()