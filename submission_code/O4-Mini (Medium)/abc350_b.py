def main():
    import sys
    data = sys.stdin.read().split()
    N, Q = map(int, data[:2])
    T = map(int, data[2:])
    # holes[i] == True means there's a tooth in hole i
    holes = [True] * (N + 1)
    for t in T:
        holes[t] = not holes[t]
    # Count teeth in holes 1 through N
    print(sum(holes[1:]))

main()