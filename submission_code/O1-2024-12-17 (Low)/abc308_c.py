def main():
    import sys
    import functools

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    people = []
    idx = 1

    for i in range(1, N + 1):
        A, B = int(data[idx]), int(data[idx + 1])
        idx += 2
        # Store tuple (id, A, B)
        people.append((i, A, B))

    # Custom comparison function for sorting
    def compare(p1, p2):
        # p1 = (id, A1, B1), p2 = (id, A2, B2)
        A1, B1 = p1[1], p1[2]
        A2, B2 = p2[1], p2[2]

        # Compare A1/(A1+B1) vs A2/(A2+B2) using cross multiplication to avoid floating issues:
        # A1 / (A1 + B1) > A2 / (A2 + B2)  ⇔  A1*(A2 + B2) > A2*(A1 + B1)
        left = A1 * (A2 + B2)
        right = A2 * (A1 + B1)

        if left > right:
            # p1's success rate is larger, so p1 should come first
            return -1
        elif left < right:
            # p2's success rate is larger, so p2 should come first
            return 1
        else:
            # Tie in success rate, compare by ascending person number
            return p1[0] - p2[0]

    # Sort using our custom comparator
    people.sort(key=functools.cmp_to_key(compare))

    # Output the sorted order by id
    print(" ".join(str(p[0]) for p in people))

# Do not forget to call main at the end
if __name__ == "__main__":
    main()