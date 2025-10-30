def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = list(map(int, data[1:]))

    p1 = P[0]
    # No need to check others if N == 1, but let's handle it consistently:
    if N == 1:
        print(0)
        return

    # Find the maximum among the other people's scores
    max_other = max(P[1:])
    
    # Calculate the required additional points for person 1
    additional_points = max(0, max_other - p1 + 1)
    print(additional_points)

# Do NOT forget to call main() at the end.
if __name__ == "__main__":
    main()