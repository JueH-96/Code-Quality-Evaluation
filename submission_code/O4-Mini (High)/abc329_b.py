def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    max_val = max(A)
    # Find the largest value strictly less than the overall maximum
    second_max = max(x for x in A if x < max_val)
    print(second_max)

main()