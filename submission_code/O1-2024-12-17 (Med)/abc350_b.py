def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])
    treatments = list(map(int, data[2:]))

    # Initialize a set of holes that contain teeth (all N holes initially).
    holes_with_teeth = set(range(1, N + 1))

    # For each treatment, toggle the tooth in the given hole.
    for hole in treatments:
        if hole in holes_with_teeth:
            holes_with_teeth.remove(hole)
        else:
            holes_with_teeth.add(hole)

    # Print the final number of teeth.
    print(len(holes_with_teeth))

# Do not forget to call the main function.
if __name__ == "__main__":
    main()