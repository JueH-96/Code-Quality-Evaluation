def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = list(map(int, data[1:]))

    # If there's only one person, they're trivially the strongest with no extra needed
    if N == 1:
        print(0)
        return

    # Person 1's current ability
    p1 = P[0]
    # The max ability among the other people
    max_other = max(P[1:])
    
    # We need x such that p1 + x > max_other
    # => x > max_other - p1
    # => x = max(0, max_other - p1 + 1)
    answer = max(0, max_other - p1 + 1)

    print(answer)

# Do not remove below line
main()