def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    names = []
    ages = []
    idx = 1
    for _ in range(N):
        names.append(data[idx])
        ages.append(int(data[idx+1]))
        idx += 2
    # find the index of the youngest person
    start = min(range(N), key=lambda i: ages[i])
    # print names in clockwise order starting from the youngest
    for i in range(N):
        print(names[(start + i) % N])

# call main to execute
main()