def main():
    import sys
    data = sys.stdin.read().strip().split()
    R, G, B = map(int, data[:3])
    C = data[3]

    if C == "Red":
        print(min(G, B))
    elif C == "Green":
        print(min(R, B))
    else:  # C == "Blue"
        print(min(R, G))

# Do not remove the function call below
main()