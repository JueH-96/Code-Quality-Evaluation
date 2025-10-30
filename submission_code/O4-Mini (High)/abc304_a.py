def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    names = []
    ages = []
    idx = 1
    for _ in range(N):
        names.append(data[idx])
        ages.append(int(data[idx+1]))
        idx += 2

    # Find the index of the youngest person
    start = min(range(N), key=lambda i: ages[i])

    # Print names in clockwise order starting from the youngest
    for k in range(N):
        i = (start + k) % N
        print(names[i])

if __name__ == "__main__":
    main()