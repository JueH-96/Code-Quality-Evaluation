def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    max_val = max(A)
    filtered = [x for x in A if x != max_val]
    # Since constraints guarantee there is at least one number != max_val
    second_max = max(filtered)
    print(second_max)

# Do not remove the call to main()!
main()