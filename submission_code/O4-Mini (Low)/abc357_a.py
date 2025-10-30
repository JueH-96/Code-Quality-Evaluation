def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    H = list(map(int, data[2:]))

    count = 0
    remaining = M

    for h in H:
        if remaining >= h:
            count += 1
            remaining -= h
        else:
            # Not enough disinfectant to cover all hands;
            # they use up what's left but don't count as successful.
            break

    print(count)

if __name__ == "__main__":
    main()