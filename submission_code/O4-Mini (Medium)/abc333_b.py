def main():
    # Map each vertex to an index around the pentagon
    idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    # Read inputs
    s = input().strip()
    t = input().strip()

    # Compute the minimal step distance on a 5-cycle
    def chord_steps(p):
        a, b = p[0], p[1]
        d = abs(idx[a] - idx[b])
        return min(d, 5 - d)

    # Compare the two chord lengths (in steps)
    if chord_steps(s) == chord_steps(t):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()